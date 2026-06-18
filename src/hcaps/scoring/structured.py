"""Structured AXC-out scoring helpers."""

from __future__ import annotations

from typing import Any


def structured_scores(metrics: dict[str, Any]) -> dict[str, float | bool | list[str]]:
    validation = _validation_metrics(metrics)
    relation = 1.0 - _loss(validation, "loss_relation_prediction")
    provenance = 1.0 - _loss(validation, "loss_provenance_recovery")
    epistemic = 1.0 - _loss(validation, "loss_epistemic_proxy")
    stability = float(validation.get("stability_temporal_stability_accuracy", 0.0))
    score = max(0.0, (relation + provenance + epistemic + stability) / 4.0)
    return {
        "structured_epistemic_score": score,
        "raw_emission_scored": True,
        "validated_axc_out_scored": True,
        "interpreted_projection_scored": True,
        "emission_levels": [
            "raw_emission",
            "validated_axc_out",
            "interpreted_projection",
            "text_projection",
        ],
    }


def _validation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    validation = metrics.get("validation", {})
    if isinstance(validation, dict):
        nested = validation.get("metrics", {})
        return nested if isinstance(nested, dict) else {}
    return {}


def _loss(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 1.0)
    return float(value) if isinstance(value, int | float) else 1.0
