"""Provenance recovery scoring metrics."""

from __future__ import annotations

from typing import Any


def provenance_scores(metrics: dict[str, Any]) -> dict[str, float]:
    values = _validation_metrics(metrics)
    return {
        "source_recall_at_1": _metric(values, "provenance_recovery_source_recall_at_1"),
        "source_recall_at_k": _metric(values, "provenance_recovery_source_recall_at_k"),
        "span_recall_at_k": _metric(values, "provenance_recovery_span_recovery_recall"),
        "provenance_mrr": _metric(values, "provenance_recovery_mean_reciprocal_rank"),
        "provenance_calibration": 0.0,
    }


def _validation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    validation = metrics.get("validation", {})
    nested = validation.get("metrics", {}) if isinstance(validation, dict) else {}
    return nested if isinstance(nested, dict) else {}


def _metric(values: dict[str, Any], key: str) -> float:
    value = values.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0
