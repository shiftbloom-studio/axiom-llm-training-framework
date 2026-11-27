"""Checkpoint save/load helpers for P6 training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct
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


# NumPy objects stored by ``capture_rng_state`` that are safe to unpickle.
_CHECKPOINT_SAFE_GLOBALS: list[Any] = [
    _reconstruct,
    np.ndarray,
    np.dtype,
    type(np.dtype(np.uint32)),
]


def load_training_checkpoint(path: str | Path, *, trusted: bool = False) -> dict[str, Any]:
    """Load a training checkpoint.

    By default only tensors, plain containers and the NumPy RNG state are unpickled
    (``weights_only=True``). Pass ``trusted=True`` to allow arbitrary pickled objects;
    only do that for checkpoints from a source you trust, since unpickling can run code.
    """

    if trusted:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    else:
        with torch.serialization.safe_globals(_CHECKPOINT_SAFE_GLOBALS):
            payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a mapping: {path}")
    return payload
