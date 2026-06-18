"""Temporal and leakage audit scoring."""

from __future__ import annotations

from typing import Any


def temporal_scores(metrics: dict[str, Any]) -> dict[str, float]:
    values = _validation_metrics(metrics)
    return {
        "temporal_holdout_loss": _metric(values, "loss_stability_temporal"),
        "temporal_stability_prediction": _metric(
            values,
            "stability_temporal_stability_accuracy",
        ),
        "outdated_belief_detection": 0.0,
        "future_target_mask_violation_count": _metric(
            values,
            "stability_temporal_future_target_mask_violation_count",
        ),
        "temporal_leakage_count": _metric(values, "stability_temporal_temporal_leakage_count"),
    }


def _validation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    validation = metrics.get("validation", {})
    nested = validation.get("metrics", {}) if isinstance(validation, dict) else {}
    return nested if isinstance(nested, dict) else {}


def _metric(values: dict[str, Any], key: str) -> float:
    value = values.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0
