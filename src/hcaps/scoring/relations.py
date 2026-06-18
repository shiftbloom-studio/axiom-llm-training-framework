"""Relation scoring metrics."""

from __future__ import annotations

from typing import Any


def relation_scores(metrics: dict[str, Any]) -> dict[str, float]:
    values = _validation_metrics(metrics)
    return {
        "relation_type_accuracy": _metric(values, "relation_prediction_relation_accuracy"),
        "relation_macro_f1": _metric(values, "relation_prediction_relation_macro_f1"),
        "relation_micro_f1": _metric(values, "relation_prediction_relation_micro_f1"),
        "support_contradiction_f1": _metric(values, "relation_prediction_relation_macro_f1"),
        "hard_negative_accuracy": _metric(values, "relation_prediction_hard_negative_accuracy"),
        "relation_target_recall_at_k": _metric(values, "relation_prediction_relation_accuracy"),
    }


def _validation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    validation = metrics.get("validation", {})
    nested = validation.get("metrics", {}) if isinstance(validation, dict) else {}
    return nested if isinstance(nested, dict) else {}


def _metric(values: dict[str, Any], key: str) -> float:
    value = values.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0
