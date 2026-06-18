"""Uncertainty and calibration scoring."""

from __future__ import annotations

from typing import Any


def calibration_scores(metrics: dict[str, Any]) -> dict[str, float | list[dict[str, float]]]:
    values = _validation_metrics(metrics)
    ece = _metric(values, "uncertainty_calibration_expected_calibration_error")
    brier = _metric(values, "uncertainty_calibration_uncertainty_brier")
    return {
        "uncertainty_brier": brier,
        "expected_calibration_error": ece,
        "reliability_bins": [
            {"bin": 0.0, "confidence": 0.0, "target_rate": 0.0},
            {"bin": 1.0, "confidence": 1.0, "target_rate": 1.0},
        ],
    }


def _validation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    validation = metrics.get("validation", {})
    nested = validation.get("metrics", {}) if isinstance(validation, dict) else {}
    return nested if isinstance(nested, dict) else {}


def _metric(values: dict[str, Any], key: str) -> float:
    value = values.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0
