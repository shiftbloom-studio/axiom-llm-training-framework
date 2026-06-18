"""Secondary text-projection scoring."""

from __future__ import annotations

import math
from typing import Any


def text_projection_scores(metrics: dict[str, Any]) -> dict[str, float]:
    validation = _validation_metrics(metrics)
    ce = _metric(validation, "text_projection_text_cross_entropy", default=0.0)
    perplexity = _metric(
        validation, "text_projection_text_perplexity", default=math.exp(min(ce, 20.0))
    )
    token_accuracy = _metric(validation, "text_projection_token_accuracy", default=0.0)
    score = max(0.0, token_accuracy + 1.0 / max(perplexity, 1.0))
    return {
        "text_cross_entropy": ce,
        "text_perplexity": perplexity,
        "token_accuracy": token_accuracy,
        "text_length_normalized_loss": ce,
        "text_projection_score": score,
    }


def _validation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    validation = metrics.get("validation", {})
    if isinstance(validation, dict):
        nested = validation.get("metrics", {})
        return nested if isinstance(nested, dict) else {}
    return {}


def _metric(metrics: dict[str, Any], key: str, *, default: float) -> float:
    value = metrics.get(key, default)
    return float(value) if isinstance(value, int | float) else default
