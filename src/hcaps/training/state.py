"""Dataclasses for P6 training state and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StepResult:
    step: int
    split: str
    total_loss: float
    metrics: dict[str, float]
    active_targets: dict[str, int]


@dataclass(frozen=True)
class TrainingRunResult:
    run_id: str
    arm_id: str
    run_dir: Path
    arm_dir: Path
    manifest_path: Path
    metrics_path: Path
    checkpoint_path: Path
    prediction_paths: dict[str, Path]
    final_step: int
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerState:
    step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    records_seen: int = 0
    tokens_seen: int = 0
    best_validation_loss: float | None = None
