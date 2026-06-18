"""P6 experiment suite orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hcaps.axt import AxtCompileConfig, compile_axt
from hcaps.experiments.arms import ArmSpec, default_arm_catalog
from hcaps.experiments.artifacts import write_reproducibility_bundle
from hcaps.experiments.budgets import ArmBudget, budget_from_metrics
from hcaps.experiments.comparison import control_effects, pairwise_metric_deltas
from hcaps.experiments.configs import ExperimentSuiteConfig
from hcaps.experiments.matching import fairness_report
from hcaps.experiments.suites import resolve_arm_ids
from hcaps.model.config import AxiomModelConfig
from hcaps.scoring import ScoringResult, score_run
from hcaps.training.config import TrainingConfig
from hcaps.training.logging import write_json
from hcaps.training.trainer import AxiomTrainer
from hcaps.verdict import VerdictThresholds, generate_verdict, write_verdict_report


@dataclass(frozen=True)
class ExperimentPlan:
    run_id: str
    run_dir: Path
    arms: list[ArmSpec]
    plan_path: Path


@dataclass(frozen=True)
class ExperimentRunResult:
    run_id: str
    run_dir: Path
    arm_results: dict[str, Any]
    scoring_result: ScoringResult
    verdict_path: Path
    fairness_report_path: Path


class ExperimentOrchestrator:
    """Expand and run P6 experiment suites sequentially."""

    def __init__(self, config: ExperimentSuiteConfig) -> None:
        self.config = config
        self.run_id = config.suite_name
        self.run_dir = config.output_dir / self.run_id
        self.catalog = default_arm_catalog(
            input_axt_path=str(config.input_axt),
            model_config=str(config.model_config_template),
            source_content_id=config.source_content_id,
            seed=config.seeds[0],
        )

    def prepare(self) -> ExperimentPlan:
        self._ensure_input_axt()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "configs" / "suite").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "configs" / "arms").mkdir(parents=True, exist_ok=True)
        arm_specs = self._expanded_arms()
        plan_payload = {
            "run_id": self.run_id,
            "suite": self.config.to_dict(),
            "arms": [arm.to_dict() for arm in arm_specs],
            "no_external_llm_training_or_evaluation": True,
        }
        plan_path = self.run_dir / "experiment_plan.json"
        write_json(plan_path, plan_payload)
        self.config.to_yaml(self.run_dir / "configs" / "suite" / "suite.yaml")
        for arm in arm_specs:
            write_json(self.run_dir / "configs" / "arms" / f"{arm.arm_id}.json", arm.to_dict())
        return ExperimentPlan(
            run_id=self.run_id, run_dir=self.run_dir, arms=arm_specs, plan_path=plan_path
        )

    def run(self) -> ExperimentRunResult:
        plan = self.prepare()
        arm_results: dict[str, Any] = {}
        budgets: list[ArmBudget] = []
        for arm in plan.arms:
            training_config = self._training_config_for_arm(arm)
            trainer = AxiomTrainer(training_config)
            result = trainer.fit()
            arm_results[arm.arm_id] = {
                "metrics_path": str(result.metrics_path),
                "checkpoint_path": str(result.checkpoint_path),
                "manifest_path": str(result.manifest_path),
            }
            budgets.append(budget_from_metrics(arm.arm_id, result.metrics))
        scoring = self.score()
        fairness = self._write_fairness_report(budgets)
        write_json(
            self.run_dir / "comparisons" / "pairwise_metrics.json",
            pairwise_metric_deltas(scoring.scores),
        )
        write_json(
            self.run_dir / "comparisons" / "control_effects.json", control_effects(scoring.scores)
        )
        verdict_path = self.verdict(scoring.scores, fairness)
        write_reproducibility_bundle(self.run_dir)
        return ExperimentRunResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            arm_results=arm_results,
            scoring_result=scoring,
            verdict_path=verdict_path,
            fairness_report_path=self.run_dir / "comparisons" / "fairness_report.json",
        )

    def score(self) -> ScoringResult:
        return score_run(self.run_dir)

    def verdict(
        self, scores: dict[str, Any] | None = None, fairness: dict[str, Any] | None = None
    ) -> Path:
        scoring = scores if scores is not None else score_run(self.run_dir).scores
        fairness_report_payload = fairness or _read_json(
            self.run_dir / "comparisons" / "fairness_report.json"
        )
        thresholds = (
            VerdictThresholds.from_yaml(self.config.verdict_config)
            if self.config.verdict_config is not None
            else VerdictThresholds()
        )
        verdict = generate_verdict(
            scoring, fairness_report=fairness_report_payload, thresholds=thresholds
        )
        return write_verdict_report(self.run_dir, verdict=verdict)

    def _expanded_arms(self) -> list[ArmSpec]:
        ids = resolve_arm_ids(self.config.arms)
        missing = [arm_id for arm_id in ids if arm_id not in self.catalog]
        if missing:
            raise ValueError(f"unknown P6 experiment arm(s): {', '.join(missing)}")
        return [self.catalog[arm_id] for arm_id in ids]

    def _training_config_for_arm(self, arm: ArmSpec) -> TrainingConfig:
        model_path = self._write_model_config_for_arm(arm)
        template = self.config.trainer_config_template
        updates = {
            "run_name": self.run_id,
            "input_axt_path": self.config.input_axt,
            "output_dir": self.config.output_dir,
            "model_config_path": model_path,
            "arm_name": arm.arm_id,
            "seed": arm.seed,
            "control_transform": arm.control_transform,
            "text_mode": None if arm.text_mode == "structured_native" else arm.text_mode,
            "geometry_config_path": self.config.geometry_config_template
            if arm.geometry_mode == "learned" or _uses_non_geometric_control(arm)
            else None,
            "geometry_provider_kind": "learned"
            if arm.geometry_mode == "learned"
            else "non_geometric_context_mixer"
            if _uses_non_geometric_control(arm)
            else "none",
            "overwrite": True,
        }
        payload = template.model_dump(mode="python")
        payload.update(updates)
        config = TrainingConfig.model_validate(payload)
        config_path = self.run_dir / "configs" / "training" / f"{arm.arm_id}.yaml"
        config.to_yaml(config_path)
        return config

    def _write_model_config_for_arm(self, arm: ArmSpec) -> Path:
        model_config = AxiomModelConfig.from_yaml(self.config.model_config_template)
        payload = model_config.model_dump(mode="python")
        ablations = {**payload.get("ablations", {}), **arm.ablations}
        payload["ablations"] = ablations
        if arm.geometry_mode == "learned" or _uses_non_geometric_control(arm):
            payload["use_geometry_features"] = True
            payload["geometry_mode"] = "geometry_provider_injected"
            payload["ablations"]["geometry_off"] = False
        else:
            payload["use_geometry_features"] = False
            payload["geometry_mode"] = "geometry_off"
            payload["ablations"]["geometry_off"] = True
        if arm.text_mode in {"flat_text", "structured_text", "capsule_text"}:
            payload["ablations"]["text_only"] = True
        config = AxiomModelConfig.model_validate(payload)
        target = self.run_dir / "configs" / "model" / f"{arm.arm_id}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), "utf-8")
        return target

    def _write_fairness_report(self, budgets: list[ArmBudget]) -> dict[str, Any]:
        report = fairness_report(
            budgets=budgets,
            source_content_id=self.config.source_content_id,
            split_id=self.config.split_id,
            extraction_substrate=str(self.config.input_axt),
            max_parameter_delta=self.config.trainer_config_template.compute_budget_config.parameter_match_tolerance,
            max_compute_delta=self.config.trainer_config_template.compute_budget_config.compute_match_tolerance,
        )
        path = self.run_dir / "comparisons" / "fairness_report.json"
        write_json(path, report)
        return report

    def _ensure_input_axt(self) -> None:
        if self.config.input_axt.exists():
            return
        if self.config.input_axp is None:
            raise FileNotFoundError(
                "configured AXT bundle is missing and no input_axp was provided: "
                f"{self.config.input_axt}"
            )
        root = Path.cwd()
        compile_axt(
            AxtCompileConfig(
                input_path=self.config.input_axp,
                output_path=self.config.input_axt,
                field_registry_path=root / "spec" / "FIELD_REGISTRY_V1.md",
                vocabulary_registry_path=root / "spec" / "VOCABULARY_REGISTRY_V1.md",
                allow_all_without_split=True,
                max_text_length=32,
            ),
            force=True,
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _uses_non_geometric_control(arm: ArmSpec) -> bool:
    return arm.arm_id == "D_structured_native_no_geometry" or arm.arm_name.endswith("no_geometry")
