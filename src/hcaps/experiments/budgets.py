"""Budget tracking and approximate FLOP estimates for P6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArmBudget:
    arm_id: str
    parameters: int
    trainable_parameters: int
    step_count: int
    records_seen: int
    tokens_seen: int
    structured_records_seen: int
    wall_clock_seconds: float
    estimated_flops: float
    hardware: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "parameters": self.parameters,
            "trainable_parameters": self.trainable_parameters,
            "step_count": self.step_count,
            "records_seen": self.records_seen,
            "tokens_seen": self.tokens_seen,
            "structured_records_seen": self.structured_records_seen,
            "wall_clock_seconds": self.wall_clock_seconds,
            "estimated_flops": self.estimated_flops,
            "hardware": self.hardware,
        }


def budget_from_metrics(arm_id: str, metrics: dict[str, Any]) -> ArmBudget:
    counts = metrics.get("parameter_count", {})
    if not isinstance(counts, dict):
        counts = {}
    parameters = int(counts.get("total", 0))
    trainable = int(counts.get("trainable", 0))
    steps = int(metrics.get("final_step", 0))
    records = int(metrics.get("records_seen", 0))
    tokens = int(metrics.get("tokens_seen", 0))
    return ArmBudget(
        arm_id=arm_id,
        parameters=parameters,
        trainable_parameters=trainable,
        step_count=steps,
        records_seen=records,
        tokens_seen=tokens,
        structured_records_seen=records,
        wall_clock_seconds=0.0,
        estimated_flops=float(2 * parameters * max(steps, 1)),
        hardware="local_runtime",
    )
