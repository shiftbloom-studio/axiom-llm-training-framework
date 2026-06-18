"""Typed configuration for P6 training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PrecisionMode = Literal["fp32", "bf16", "fp16"]
OptimizerName = Literal["adamw", "sgd"]
SchedulerName = Literal["none", "linear_warmup_decay"]
CurriculumMode = Literal["none", "linear_complexity_ramp", "staged_complexity_ramp"]
GeometryProviderKind = Literal["none", "learned", "non_geometric_context_mixer"]


class LossWeights(BaseModel):
    """Explicit P6 loss weights.

    Text projection remains mandatory but secondary. Geometry weights regularize
    the P5 module without encoding a thesis as a target law.
    """

    model_config = ConfigDict(extra="forbid")

    structured_axc_out: float = 1.0
    relation_prediction: float = 1.0
    provenance_recovery: float = 1.0
    epistemic_proxy: float = 1.0
    stability_temporal: float = 1.0
    uncertainty_calibration: float = 0.5
    geometry_observables: float = 0.2
    geometry_regularization: float = 0.01
    text_projection: float = 0.2


class CurriculumConfig(BaseModel):
    """Training-time complexity ramp over objective exposure and weights."""

    model_config = ConfigDict(extra="forbid")

    schedule_name: CurriculumMode = "none"
    phase_boundaries: list[int] = Field(default_factory=lambda: [0, 10, 25, 50, 75, 100])
    text_projection_weight_schedule: list[float] = Field(default_factory=lambda: [1.0])
    structured_loss_weight_schedule: list[float] = Field(default_factory=lambda: [1.0])
    relation_neighborhood_depth_schedule: list[int] = Field(default_factory=lambda: [1])
    side_channel_dropout_schedule: list[float] = Field(default_factory=lambda: [0.0])
    geometry_activation_schedule: list[float] = Field(default_factory=lambda: [1.0])
    context_dropout_schedule: list[float] = Field(default_factory=lambda: [0.0])
    negative_sample_hardness_schedule: list[float] = Field(default_factory=lambda: [0.0])


class ComputeBudgetConfig(BaseModel):
    """Bounded compute budget and reporting tolerances."""

    model_config = ConfigDict(extra="forbid")

    max_train_steps: int = Field(default=2, ge=1)
    max_records_seen: int | None = Field(default=None, ge=1)
    max_wall_clock_seconds: float | None = Field(default=None, gt=0.0)
    parameter_match_tolerance: float = Field(default=0.05, ge=0.0)
    compute_match_tolerance: float = Field(default=0.10, ge=0.0)
    estimate_flops: bool = True


class TrainingConfig(BaseModel):
    """Configuration for one arm-level P6 training run."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    seed: int
    input_axt_path: Path
    output_dir: Path
    model_config_path: Path
    arm_name: str
    train_split: str | None = None
    validation_split: str | None = None
    max_steps: int = Field(default=2, ge=1)
    max_epochs: int = Field(default=1, ge=1)
    global_batch_size: int = Field(default=2, ge=1)
    micro_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "none"
    warmup_steps: int = Field(default=0, ge=0)
    cooldown_steps: int = Field(default=0, ge=0)
    clip_grad_norm: float | None = Field(default=1.0, gt=0.0)
    precision: PrecisionMode = "fp32"
    checkpoint_interval: int = Field(default=1, ge=1)
    eval_interval: int = Field(default=1, ge=1)
    log_interval: int = Field(default=1, ge=1)
    save_optimizer_state: bool = True
    resume_from: Path | None = None
    structured_loss_weights: LossWeights = Field(default_factory=LossWeights)
    text_projection_loss_weight: float = Field(default=0.2, ge=0.0)
    geometry_loss_weight: float = Field(default=0.2, ge=0.0)
    curriculum_config: CurriculumConfig = Field(default_factory=CurriculumConfig)
    compute_budget_config: ComputeBudgetConfig = Field(default_factory=ComputeBudgetConfig)
    geometry_config_path: Path | None = None
    geometry_provider_kind: GeometryProviderKind = "none"
    control_transform: str | None = None
    text_mode: str | None = None
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_training(self) -> Self:
        if self.global_batch_size % self.micro_batch_size != 0:
            raise ValueError("global_batch_size must be divisible by micro_batch_size")
        if self.seed is None:
            raise ValueError("seed must be explicit")
        if self.text_projection_loss_weight != self.structured_loss_weights.text_projection:
            updated = self.structured_loss_weights.model_copy(
                update={"text_projection": self.text_projection_loss_weight}
            )
            object.__setattr__(self, "structured_loss_weights", updated)
        if self.geometry_loss_weight != self.structured_loss_weights.geometry_observables:
            updated = self.structured_loss_weights.model_copy(
                update={"geometry_observables": self.geometry_loss_weight}
            )
            object.__setattr__(self, "structured_loss_weights", updated)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"training config must be a YAML mapping: {path}")
        return cls.model_validate(payload)

    def to_yaml(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True), "utf-8")

    def run_id(self) -> str:
        return self.run_name

    def run_dir(self) -> Path:
        return self.output_dir / self.run_id()

    def arm_dir(self) -> Path:
        return self.run_dir() / "arms" / self.arm_name

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
