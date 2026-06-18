from __future__ import annotations

from tests.spec._helpers import assert_contains_all, read_spec


def test_loss_and_target_masks_are_complete() -> None:
    text = read_spec("LOSS_AND_TARGET_MASKS_V1.md")

    assert_contains_all(
        text,
        [
            "loss_mask_text_projection",
            "loss_mask_relation_prediction",
            "loss_mask_provenance_recovery",
            "loss_mask_stability_prediction",
            "loss_mask_uncertainty_calibration",
            "loss_mask_future_summary",
            "loss_mask_temporal_prediction",
            "loss_mask_geometry_observables",
            "loss_mask_provider_context",
            "loss_mask_negative_sampling",
            "available_relation_targets",
            "available_provenance_targets",
            "available_epistemic_targets",
            "available_future_targets",
            "available_geometry_targets",
            "available_text_targets",
            "available_evaluation_references",
        ],
    )


def test_mask_states_distinguish_missing_from_negative() -> None:
    text = read_spec("LOSS_AND_TARGET_MASKS_V1.md")

    assert_contains_all(
        text,
        [
            "unknown",
            "missing",
            "not_applicable",
            "not_predictor_visible",
            "target_only",
            "masked_by_split",
            "available",
            "Missing target availability must never be interpreted as a negative target",
        ],
    )
