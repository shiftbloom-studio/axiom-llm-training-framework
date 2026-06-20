"""P6 trainer for P4/P5-compatible structured-native Axiom models."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hcaps.axt import AxtBatch, AxtBatchCollator, AxtDataset
from hcaps.experiments.controls import apply_control_transform, control_manifest
from hcaps.format.hashing import canonical_json_bytes, file_hash, sha256_bytes
from hcaps.geometry import GeometryConfig
from hcaps.geometry.module import P4GeometryProvider, P4NonGeometricContextProvider
from hcaps.model import (
    AxiomModelConfig,
    AxiomModelInput,
    AxiomStructuredModel,
    StructuredInputAdapter,
)
from hcaps.model.axc_out import AXCOutEmission
from hcaps.model.geometry_hooks import GeometryProviderProtocol
from hcaps.model.parameter_count import parameter_count
from hcaps.training.checkpointing import (
    LOSS_REGISTRY_VERSION,
    axt_manifest_hash,
    load_training_checkpoint,
    model_config_hash,
    save_training_checkpoint,
    training_config_hash,
)
from hcaps.training.config import TrainingConfig
from hcaps.training.curriculum import curriculum_multipliers
from hcaps.training.logging import (
    append_jsonl,
    environment_summary,
    git_state_summary,
    write_json,
)
from hcaps.training.loss_registry import LossRegistry, build_default_loss_registry
from hcaps.training.optim import build_optimizer, build_scheduler
from hcaps.training.precision import autocast_context
from hcaps.training.seeds import restore_rng_state, set_global_seed
from hcaps.training.state import StepResult, TrainerState, TrainingRunResult
from hcaps.utils.time import utc_now

MIN_LOGIT_RANK = 2


class AxiomTrainer:
    """Train one P6 arm from an AXT bundle through native AXC-out losses."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        loss_registry: LossRegistry | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        set_global_seed(config.seed)
        self.model_config = AxiomModelConfig.from_yaml(config.model_config_path)
        self._assert_geometry_intent_realized()
        self.model = self._build_model().to(self.device)
        self.input_adapter = StructuredInputAdapter(self.model_config, device=self.device)
        self.loss_registry = loss_registry or build_default_loss_registry(
            config.structured_loss_weights
        )
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        self.state = TrainerState()
        self.collator = AxtBatchCollator()
        self._prepare_directories()
        if config.resume_from is not None:
            self.load_checkpoint(config.resume_from)

    def fit(self) -> TrainingRunResult:
        """Run training, validation, checkpointing, prediction, and manifests."""

        train_dataset = AxtDataset(self.config.input_axt_path, split_name=self.config.train_split)
        if len(train_dataset) == 0:
            raise ValueError("training split contains no AXT records")
        validation_dataset = AxtDataset(
            self.config.input_axt_path,
            split_name=self.config.validation_split,
        )
        final_checkpoint: Path | None = None
        while self.state.step < self.config.max_steps:
            for batch in self._batches(train_dataset):
                result = self.train_step(batch)
                if self.state.step % self.config.log_interval == 0:
                    append_jsonl(self.config.arm_dir() / "train_log.jsonl", _step_payload(result))
                if self.state.step % self.config.eval_interval == 0:
                    validation = self.validation_step(self._first_batch(validation_dataset))
                    append_jsonl(
                        self.config.arm_dir() / "validation_log.jsonl",
                        _step_payload(validation),
                    )
                if self.state.step % self.config.checkpoint_interval == 0:
                    final_checkpoint = self.save_checkpoint(self.state.step)
                if self.state.step >= self.config.max_steps:
                    break
            self.state.epoch += 1
            if (
                self.state.epoch >= self.config.max_epochs
                and self.state.step >= self.config.max_steps
            ):
                break
        if final_checkpoint is None:
            final_checkpoint = self.save_checkpoint(self.state.step)

        metrics = self._write_metrics()
        prediction_paths = self._write_predictions(validation_dataset)
        manifest_path = self._write_manifests(metrics, prediction_paths, final_checkpoint)
        return TrainingRunResult(
            run_id=self.config.run_id(),
            arm_id=self.config.arm_name,
            run_dir=self.config.run_dir(),
            arm_dir=self.config.arm_dir(),
            manifest_path=manifest_path,
            metrics_path=self.config.arm_dir() / "metrics.json",
            checkpoint_path=final_checkpoint,
            prediction_paths=prediction_paths,
            final_step=self.state.step,
            metrics=metrics,
        )

    def train_step(self, batch: AxtBatch) -> StepResult:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        transformed = apply_control_transform(
            batch,
            control_transform=self.config.control_transform,
            text_mode=self.config.text_mode,
        )
        model_input = self.input_adapter(transformed)
        with autocast_context(self.device, self.config.precision):
            output = self.model(model_input)
            loss_result = self.loss_registry.compute(output=output, model_input=model_input)
            multipliers = curriculum_multipliers(
                self.config.curriculum_config,
                step=self.state.step,
                max_steps=self.config.max_steps,
            )
            loss = _apply_curriculum(loss_result.total_loss, multipliers)
        loss.backward()  # type: ignore[no-untyped-call]
        if self.config.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.state.step += 1
        self.state.samples_seen += batch.size
        self.state.records_seen += batch.size
        self.state.tokens_seen += _text_token_count(model_input)
        return StepResult(
            step=self.state.step,
            split="train",
            total_loss=float(loss.detach().cpu().item()),
            metrics=loss_result.metrics(),
            active_targets=loss_result.active_targets(),
        )

    def validation_step(self, batch: AxtBatch) -> StepResult:
        self.model.eval()
        transformed = apply_control_transform(
            batch,
            control_transform=self.config.control_transform,
            text_mode=self.config.text_mode,
        )
        with torch.no_grad(), autocast_context(self.device, self.config.precision):
            model_input = self.input_adapter(transformed)
            output = self.model(model_input)
            loss_result = self.loss_registry.compute(output=output, model_input=model_input)
        return StepResult(
            step=self.state.step,
            split="validation",
            total_loss=float(loss_result.total_loss.detach().cpu().item()),
            metrics=loss_result.metrics(),
            active_targets=loss_result.active_targets(),
        )

    def save_checkpoint(self, step: int) -> Path:
        checkpoint_path = self.config.arm_dir() / "checkpoints" / f"step_{step:06d}.pt"
        return save_training_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            state=self.state,
            training_config=self.config,
            model_config=self.model_config,
            save_optimizer_state=self.config.save_optimizer_state,
            curriculum_state=curriculum_multipliers(
                self.config.curriculum_config,
                step=max(0, step - 1),
                max_steps=self.config.max_steps,
            ),
        )

    def load_checkpoint(self, path: Path) -> None:
        payload = load_training_checkpoint(path)
        self.model.load_state_dict(payload["model_state"])
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        scheduler_state = payload.get("scheduler_state")
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)
        self.state.step = int(payload.get("step", 0))
        self.state.epoch = int(payload.get("epoch", 0))
        rng_state = payload.get("rng_state")
        if isinstance(rng_state, dict):
            restore_rng_state(rng_state)

    def _build_model(self) -> AxiomStructuredModel:
        geometry_provider: GeometryProviderProtocol | None = None
        if self.config.geometry_config_path is not None:
            geometry_config = GeometryConfig.from_yaml(self.config.geometry_config_path)
            if self.config.geometry_provider_kind == "non_geometric_context_mixer":
                geometry_provider = P4NonGeometricContextProvider(
                    geometry_config,
                    model_dim=self.model_config.model_dim,
                )
            elif geometry_config.mode.value != "off":
                geometry_provider = P4GeometryProvider(
                    geometry_config,
                    model_dim=self.model_config.model_dim,
                )
        return AxiomStructuredModel(self.model_config, geometry_provider=geometry_provider)

    def _assert_geometry_intent_realized(self) -> None:
        """Fail loudly when geometry is configured but would not actually run.

        The training config only BUILDS a geometry provider; the model config's
        effective_geometry_mode is what CALLS it. If they disagree, arm E would
        silently train as arm D (no geometry). Raise instead of degrading.
        """
        provider_supplied = (
            self.config.geometry_provider_kind != "none"
            and self.config.geometry_config_path is not None
        )
        wants_learned = (
            self.config.geometry_provider_kind == "learned"
            and self.config.geometry_config_path is not None
        )
        effective_mode = self.model_config.effective_geometry_mode()
        injected = effective_mode == "geometry_provider_injected"
        if wants_learned and not injected:
            raise ValueError(
                "Geometry misconfiguration: training config requests learned geometry "
                "(geometry_provider_kind='learned', geometry_config_path="
                f"'{self.config.geometry_config_path}') but model config "
                f"'{self.config.model_config_path}' resolves to effective_geometry_mode="
                f"'{effective_mode}'. The learned transport module would be built but never "
                "called -- arm E would silently degrade to arm D. Set the model config "
                "geometry.mode='geometry_provider_injected' (enabled: true, "
                "ablations.geometry_off: false)."
            )
        if injected and not provider_supplied:
            raise ValueError(
                "Geometry misconfiguration: model config "
                f"'{self.config.model_config_path}' resolves to effective_geometry_mode="
                "'geometry_provider_injected' but the training config supplies no provider to "
                f"inject (geometry_provider_kind='{self.config.geometry_provider_kind}', "
                f"geometry_config_path='{self.config.geometry_config_path}'). Set "
                "geometry_provider_kind to 'learned' (or 'non_geometric_context_mixer') and a "
                "geometry_config_path."
            )

    def _prepare_directories(self) -> None:
        arm_dir = self.config.arm_dir()
        if arm_dir.exists() and not self.config.overwrite and self.config.resume_from is None:
            raise FileExistsError(
                f"training arm directory already exists; pass overwrite/resume: {arm_dir}"
            )
        for path in (
            self.config.run_dir(),
            arm_dir / "checkpoints",
            arm_dir / "predictions",
            self.config.run_dir() / "configs" / "training",
            self.config.run_dir() / "configs" / "model",
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(
            self.config.run_dir() / "configs" / "training" / f"{self.config.arm_name}.yaml"
        )

    def _batches(self, dataset: AxtDataset) -> list[AxtBatch]:
        batches: list[AxtBatch] = []
        records = [dataset[index] for index in range(len(dataset))]
        for start in range(0, len(records), self.config.micro_batch_size):
            batches.append(self.collator(records[start : start + self.config.micro_batch_size]))
        return batches

    def _first_batch(self, dataset: AxtDataset) -> AxtBatch:
        if len(dataset) == 0:
            raise ValueError("validation split contains no AXT records")
        count = min(self.config.micro_batch_size, len(dataset))
        return self.collator([dataset[index] for index in range(count)])

    def _write_metrics(self) -> dict[str, Any]:
        train_metrics = _read_last_jsonl(self.config.arm_dir() / "train_log.jsonl")
        validation_metrics = _read_last_jsonl(self.config.arm_dir() / "validation_log.jsonl")
        metrics = {
            "run_id": self.config.run_id(),
            "arm_id": self.config.arm_name,
            "final_step": self.state.step,
            "records_seen": self.state.records_seen,
            "tokens_seen": self.state.tokens_seen,
            "train": train_metrics,
            "validation": validation_metrics,
            "parameter_count": parameter_count(self.model),
            "loss_registry_version": LOSS_REGISTRY_VERSION,
            "no_benchmark_claim": True,
        }
        write_json(self.config.arm_dir() / "metrics.json", metrics)
        return metrics

    def _write_predictions(self, dataset: AxtDataset) -> dict[str, Path]:
        prediction_dir = self.config.arm_dir() / "predictions"
        paths = {
            "raw_emission": prediction_dir / "raw_emission.jsonl",
            "validated_axc_out": prediction_dir / "validated_axc_out.jsonl",
            "interpreted_projection": prediction_dir / "interpreted_projection.jsonl",
            "text_projection": prediction_dir / "text_projection.jsonl",
        }
        for path in paths.values():
            path.write_text("", encoding="utf-8")
        self.model.eval()
        with torch.no_grad():
            for batch in self._batches(dataset):
                transformed = apply_control_transform(
                    batch,
                    control_transform=self.config.control_transform,
                    text_mode=self.config.text_mode,
                )
                model_input = self.input_adapter(transformed)
                output = self.model(model_input)
                self._append_prediction_records(output.raw_axc_out, paths)
        return paths

    def _append_prediction_records(
        self,
        emission: AXCOutEmission,
        paths: dict[str, Path],
    ) -> None:
        raw = emission.raw_emission
        batch_size = raw.claim_state_logits.shape[0]
        for index in range(batch_size):
            raw_payload = {
                "emission_level": "raw_emission",
                "record_offset": index,
                "head_shapes": raw.head_shapes(),
                "top_candidates": _top_candidates(raw.tensor_fields(), index),
                "raw_emission_preserved": True,
            }
            append_jsonl(paths["raw_emission"], raw_payload)
            append_jsonl(
                paths["validated_axc_out"],
                {"emission_level": "validated_axc_out", **emission.validated_axc_out},
            )
            append_jsonl(
                paths["interpreted_projection"],
                {"emission_level": "interpreted_projection", **emission.interpreted_projection},
            )
            append_jsonl(
                paths["text_projection"],
                {"emission_level": "text_projection", **emission.text_projection},
            )

    def _write_manifests(
        self,
        metrics: dict[str, Any],
        prediction_paths: dict[str, Path],
        checkpoint_path: Path,
    ) -> Path:
        repo_root = Path.cwd()
        input_artifacts = self._write_input_artifacts()
        arm_manifest = {
            "run_id": self.config.run_id(),
            "arm_id": self.config.arm_name,
            "input_axt_path": str(self.config.input_axt_path),
            "axt_manifest_hash": axt_manifest_hash(self.config.input_axt_path),
            "model_config_hash": model_config_hash(self.model_config),
            "training_config_hash": training_config_hash(self.config),
            "loss_config_hash": sha256_bytes(
                canonical_json_bytes(self.config.structured_loss_weights.model_dump(mode="json"))
            ),
            "loss_registry_version": LOSS_REGISTRY_VERSION,
            "seed": self.config.seed,
            "created_at": utc_now().isoformat(),
            "control": control_manifest(self.config.control_transform, self.config.text_mode),
            "checkpoint_path": str(checkpoint_path),
            "prediction_hashes": {
                name: file_hash(path) for name, path in prediction_paths.items() if path.exists()
            },
            "metrics": metrics,
            "provider_calls_allowed": False,
        }
        arm_manifest_path = self.config.arm_dir() / "arm_manifest.json"
        write_json(arm_manifest_path, arm_manifest)
        run_manifest_path = self.config.run_dir() / "run_manifest.json"
        run_manifest = {
            "run_id": self.config.run_id(),
            "created_at": utc_now().isoformat(),
            "input": {
                "axt_path": str(self.config.input_axt_path),
                "axt_manifest_hash": axt_manifest_hash(self.config.input_axt_path),
                "artifacts": input_artifacts,
            },
            "environment": environment_summary(),
            "git_state": git_state_summary(repo_root),
            "arms": {self.config.arm_name: arm_manifest},
            "artifact_layout": "P6",
            "no_external_llm_training_or_evaluation": True,
        }
        if run_manifest_path.exists():
            existing = _read_json(run_manifest_path)
            arms = existing.get("arms", {})
            if isinstance(arms, dict):
                arms[self.config.arm_name] = arm_manifest
                existing["arms"] = arms
                run_manifest = existing
        write_json(run_manifest_path, run_manifest)
        write_json(self.config.run_dir() / "environment.json", environment_summary())
        write_json(self.config.run_dir() / "git_state.json", git_state_summary(repo_root))
        return arm_manifest_path

    def _write_input_artifacts(self) -> dict[str, str]:
        input_dir = self.config.run_dir() / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        copied: dict[str, str] = {}
        sources = {
            "axt_manifest": self.config.input_axt_path / "axiom.json",
            "dataset_hashes": self.config.input_axt_path / "manifests" / "hashes.json",
            "source_index": self.config.input_axt_path / "manifests" / "source_index.json",
            "tensor_manifest": self.config.input_axt_path / "manifests" / "tensor_manifest.json",
            "mask_report": self.config.input_axt_path / "reports" / "mask_report.json",
            "missing_target_report": self.config.input_axt_path
            / "reports"
            / "missing_target_report.json",
        }
        for name, source in sources.items():
            if not source.exists():
                continue
            target = input_dir / source.name
            shutil.copy2(source, target)
            copied[name] = str(target.relative_to(self.config.run_dir()))
        return copied


def _apply_curriculum(loss: torch.Tensor, multipliers: dict[str, float]) -> torch.Tensor:
    multiplier = float(multipliers.get("structured", 1.0))
    return loss * multiplier


def _text_token_count(model_input: AxiomModelInput) -> int:
    text = model_input.group("text_projection")
    mask = text.get("text_loss_mask")
    if mask is None:
        mask = text.get("text_attention_mask")
    if mask is None:
        return 0
    return int(mask.to(dtype=torch.long).sum().detach().cpu().item())


def _step_payload(result: StepResult) -> dict[str, Any]:
    return {
        "step": result.step,
        "split": result.split,
        "total_loss": result.total_loss,
        "metrics": result.metrics,
        "active_targets": result.active_targets,
        "created_at": utc_now().isoformat(),
    }


def _read_last_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    payload = json.loads(lines[-1])
    return payload if isinstance(payload, dict) else {}


def _top_candidates(fields: dict[str, torch.Tensor], index: int) -> dict[str, Any]:
    top: dict[str, Any] = {}
    for name, tensor in fields.items():
        if tensor.ndim < MIN_LOGIT_RANK:
            continue
        record = tensor[index].reshape(-1)
        if record.numel() == 0:
            continue
        value, candidate_index = torch.max(record, dim=0)
        top[name] = {
            "index": int(candidate_index.detach().cpu().item()),
            "score": float(value.detach().cpu().item()),
        }
    return top


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
