"""Geometry evidence metrics for P6."""

from __future__ import annotations

from typing import Any


def geometry_scores(metrics: dict[str, Any]) -> dict[str, float]:
    values = _validation_metrics(metrics)
    observable_loss = _metric(values, "geometry_auxiliary_geometry_observable_loss")
    regularization = _metric(values, "geometry_auxiliary_geometry_regularization")
    evidence = max(0.0, 1.0 - observable_loss - 0.01 * regularization)
    return {
        "geometry_evidence_score": evidence,
        "geometry_observable_stability": evidence,
        "transport_consistency_score": max(0.0, 1.0 - regularization),
        "curvature_observable_correlation_with_revision": 0.0,
    }


def _validation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    validation = metrics.get("validation", {})
    nested = validation.get("metrics", {}) if isinstance(validation, dict) else {}
    return nested if isinstance(nested, dict) else {}


def _metric(values: dict[str, Any], key: str) -> float:
    value = values.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0
