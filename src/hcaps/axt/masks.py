"""Availability and loss-mask compilation."""

from __future__ import annotations

from typing import Any

from .schema import AVAILABILITY_MASK_NAMES, LOSS_MASK_NAMES, TensorGroup, as_i64


def compile_availability_and_loss_masks(
    capsules: list[Any],
    *,
    provider_context_mask: list[int],
    negative_sample_mask: list[int],
    evaluation_reference_count: int,
) -> tuple[TensorGroup, TensorGroup, dict[str, int], dict[str, int]]:
    """Compile P2 availability and loss-mask concepts."""

    availability: dict[str, list[int]] = {name: [] for name in AVAILABILITY_MASK_NAMES}
    loss: dict[str, list[int]] = {name: [] for name in LOSS_MASK_NAMES}

    for index, capsule in enumerate(capsules):
        relation_available = int(bool(capsule.relations))
        provenance_available = int(bool(capsule.provenance.sources))
        epistemic_available = int(_has_epistemic_targets(capsule))
        future_available = int(bool(capsule.training.future_label_fields))
        geometry_available = int(
            bool(capsule.geometry.enabled and capsule.geometry.curvature_score is not None)
        )
        text_available = int(bool(capsule.claim.canonical_text))
        evaluation_available = int(evaluation_reference_count > 0)
        negative_available = negative_sample_mask[index] if index < len(negative_sample_mask) else 0
        provider_available = (
            provider_context_mask[index] if index < len(provider_context_mask) else 0
        )

        availability["available_relation_targets"].append(relation_available)
        availability["available_provenance_targets"].append(provenance_available)
        availability["available_epistemic_targets"].append(epistemic_available)
        availability["available_future_targets"].append(future_available)
        availability["available_geometry_targets"].append(geometry_available)
        availability["available_text_targets"].append(text_available)
        availability["available_evaluation_references"].append(evaluation_available)

        loss["loss_mask_text_projection"].append(text_available)
        loss["loss_mask_relation_prediction"].append(relation_available)
        loss["loss_mask_provenance_recovery"].append(provenance_available)
        loss["loss_mask_stability_prediction"].append(epistemic_available)
        loss["loss_mask_uncertainty_calibration"].append(
            int(capsule.epistemic_state.uncertainty is not None)
        )
        loss["loss_mask_future_summary"].append(future_available)
        loss["loss_mask_temporal_prediction"].append(1)
        loss["loss_mask_geometry_observables"].append(geometry_available)
        loss["loss_mask_provider_context"].append(provider_available)
        loss["loss_mask_negative_sampling"].append(negative_available)

    availability_group = {name: as_i64(values) for name, values in availability.items()}
    loss_group = {name: as_i64(values) for name, values in loss.items()}
    return (
        availability_group,
        loss_group,
        {name: sum(values) for name, values in availability.items()},
        {name: sum(values) for name, values in loss.items()},
    )


def _has_epistemic_targets(capsule: Any) -> bool:
    epistemic = capsule.epistemic_state
    return any(
        value is not None
        for value in (
            epistemic.ontology_compatibility,
            epistemic.evidential_anchoring,
            epistemic.transformation_pressure,
            epistemic.uncertainty,
            epistemic.independent_redundancy,
        )
    )
