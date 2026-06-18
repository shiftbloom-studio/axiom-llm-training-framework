"""Checkpoint save/load helpers for P6 training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hcaps.format.hashing import canonical_json_bytes, file_hash, sha256_bytes
from hcaps.model.config import AxiomModelConfig
from hcaps.model.parameter_count import parameter_count
from hcaps.training.config import TrainingConfig
from hcaps.training.seeds import capture_rng_state
from hcaps.training.state import TrainerState
from hcaps.utils.time import utc_now

LOSS_REGISTRY_VERSION = "p6-loss-registry-0.1.0"


def training_config_hash(config: TrainingConfig) -> str:
    return sha256_bytes(canonical_json_bytes(config.hash_payload()))


def model_config_hash(config: AxiomModelConfig) -> str:
    return sha256_bytes(canonical_json_bytes(config.model_dump(mode="json")))


def axt_manifest_hash(input_axt_path: Path) -> str:
    return file_hash(input_axt_path / "axiom.json")


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    state: TrainerState,
    training_config: TrainingConfig,
    model_config: AxiomModelConfig,
    save_optimizer_state: bool,
    curriculum_state: dict[str, Any],
) -> Path:
    """Persist model, optimizer, scheduler, RNG, and manifest hashes."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if save_optimizer_state else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": state.step,
        "epoch": state.epoch,
        "rng_state": capture_rng_state(),
        "training_config": training_config.model_dump(mode="json"),
        "model_config": model_config.model_dump(mode="json"),
        "training_config_hash": training_config_hash(training_config),
        "axt_input_manifest_hash": axt_manifest_hash(training_config.input_axt_path),
        "model_config_hash": model_config_hash(model_config),
        "loss_registry_version": LOSS_REGISTRY_VERSION,
        "curriculum_state": curriculum_state,
        "parameter_count": parameter_count(model),
        "created_at": utc_now().isoformat(),
    }
    torch.save(payload, target)
    latest = target.parent / "latest.pt"
    torch.save(payload, latest)
    return target


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a mapping: {path}")
    return payload
