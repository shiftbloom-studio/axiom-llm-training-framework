"""Named suite helpers for P6."""

from __future__ import annotations

from hcaps.experiments.arms import smoke_arm_ids


def resolve_arm_ids(requested: list[str]) -> tuple[str, ...]:
    if not requested or requested == ["smoke"]:
        return smoke_arm_ids()
    if requested == ["all"]:
        return (
            "A_flat_text",
            "B_structured_text",
            "C_capsule_text",
            "D_structured_native_no_geometry",
            "E_structured_native_geometry",
            "F_structured_native_no_provenance",
            "G_structured_native_no_relations",
            "H_structured_native_context_shuffle",
            "I_structured_native_provider_shuffle",
            "J_popularity_frequency_control",
        )
    return tuple(requested)
