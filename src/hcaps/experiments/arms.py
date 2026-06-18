"""Experiment arm specifications for P6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

GeometryArmMode = Literal["off", "learned", "precomputed", "reference"]
TextMode = Literal["flat_text", "structured_text", "capsule_text", "structured_native"]


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    arm_name: str
    source_content_id: str
    input_axt_path: str
    model_config: str
    loss_config: dict[str, float] = field(default_factory=dict)
    geometry_mode: GeometryArmMode = "off"
    text_mode: TextMode = "structured_native"
    control_transform: str | None = None
    parameter_budget: str = "matched_smoke"
    compute_budget: str = "matched_smoke"
    training_schedule: str = "matched_smoke"
    seed: int = 13
    expected_outputs: tuple[str, ...] = (
        "checkpoint",
        "metrics",
        "predictions",
        "manifest",
    )
    ablations: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "arm_name": self.arm_name,
            "source_content_id": self.source_content_id,
            "input_axt_path": self.input_axt_path,
            "model_config": self.model_config,
            "loss_config": self.loss_config,
            "geometry_mode": self.geometry_mode,
            "text_mode": self.text_mode,
            "control_transform": self.control_transform,
            "parameter_budget": self.parameter_budget,
            "compute_budget": self.compute_budget,
            "training_schedule": self.training_schedule,
            "seed": self.seed,
            "expected_outputs": list(self.expected_outputs),
            "ablations": self.ablations,
        }


def default_arm_catalog(
    *,
    input_axt_path: str,
    model_config: str,
    source_content_id: str = "same_source_content",
    seed: int = 13,
) -> dict[str, ArmSpec]:
    """Return the full P6 arm catalog."""

    def arm(**kwargs: Any) -> ArmSpec:
        payload = {
            "input_axt_path": input_axt_path,
            "model_config": model_config,
            "source_content_id": source_content_id,
            "seed": seed,
            **kwargs,
        }
        return ArmSpec(**payload)

    return {
        "A_flat_text": arm(
            arm_id="A_flat_text",
            arm_name="flat_text",
            text_mode="flat_text",
            ablations={"text_only": True, "geometry_off": True},
            geometry_mode="off",
        ),
        "B_structured_text": arm(
            arm_id="B_structured_text",
            arm_name="structured_text",
            text_mode="structured_text",
            ablations={"text_only": True, "geometry_off": True},
            geometry_mode="off",
        ),
        "C_capsule_text": arm(
            arm_id="C_capsule_text",
            arm_name="capsule_text",
            text_mode="capsule_text",
            ablations={"text_only": True, "geometry_off": True},
            geometry_mode="off",
        ),
        "D_structured_native_no_geometry": arm(
            arm_id="D_structured_native_no_geometry",
            arm_name="structured_native_no_geometry",
            geometry_mode="off",
            ablations={"geometry_off": True},
        ),
        "E_structured_native_geometry": arm(
            arm_id="E_structured_native_geometry",
            arm_name="structured_native_geometry",
            geometry_mode="learned",
            ablations={"geometry_off": False},
        ),
        "F_structured_native_no_provenance": arm(
            arm_id="F_structured_native_no_provenance",
            arm_name="structured_native_no_provenance",
            geometry_mode="off",
            control_transform="no_provenance",
            ablations={"no_provenance": True, "geometry_off": True},
        ),
        "G_structured_native_no_relations": arm(
            arm_id="G_structured_native_no_relations",
            arm_name="structured_native_no_relations",
            geometry_mode="off",
            control_transform="no_relations",
            ablations={"no_relations": True, "geometry_off": True},
        ),
        "H_structured_native_context_shuffle": arm(
            arm_id="H_structured_native_context_shuffle",
            arm_name="structured_native_context_shuffle",
            geometry_mode="learned",
            control_transform="context_shuffle",
            ablations={"geometry_off": False},
        ),
        "I_structured_native_provider_shuffle": arm(
            arm_id="I_structured_native_provider_shuffle",
            arm_name="structured_native_provider_shuffle",
            geometry_mode="learned",
            control_transform="provider_shuffle",
            ablations={"geometry_off": False},
        ),
        "J_popularity_frequency_control": arm(
            arm_id="J_popularity_frequency_control",
            arm_name="popularity_frequency_control",
            geometry_mode="off",
            control_transform="popularity_frequency_control",
            ablations={"geometry_off": True},
        ),
    }


def smoke_arm_ids() -> tuple[str, ...]:
    return (
        "A_flat_text",
        "B_structured_text",
        "D_structured_native_no_geometry",
        "E_structured_native_geometry",
        "H_structured_native_context_shuffle",
        "J_popularity_frequency_control",
    )
