"""Pairwise and control-effect comparison helpers."""

from __future__ import annotations

from typing import Any


def pairwise_metric_deltas(scores: dict[str, Any]) -> dict[str, Any]:
    arms = scores.get("arms", {})
    if not isinstance(arms, dict):
        return {}
    output: dict[str, Any] = {}
    arm_items = list(arms.items())
    for left_id, left in arm_items:
        for right_id, right in arm_items:
            if left_id >= right_id or not isinstance(left, dict) or not isinstance(right, dict):
                continue
            output[f"{left_id}__vs__{right_id}"] = {
                "structured_epistemic_delta": _metric(right, "structured_epistemic_score")
                - _metric(left, "structured_epistemic_score"),
                "text_projection_delta": _metric(right, "text_projection_score")
                - _metric(left, "text_projection_score"),
                "geometry_evidence_delta": _metric(right, "geometry_evidence_score")
                - _metric(left, "geometry_evidence_score"),
            }
    return output


def control_effects(scores: dict[str, Any]) -> dict[str, float]:
    arms = scores.get("arms", {})
    if not isinstance(arms, dict):
        return {}
    geometry = _metric_dict(arms, "E_structured_native_geometry", "structured_epistemic_score")
    context = _metric_dict(
        arms, "H_structured_native_context_shuffle", "structured_epistemic_score"
    )
    no_geometry = _metric_dict(
        arms, "D_structured_native_no_geometry", "structured_epistemic_score"
    )
    popularity = _metric_dict(arms, "J_popularity_frequency_control", "structured_epistemic_score")
    return {
        "geometry_on_minus_off_delta": geometry - no_geometry,
        "context_shuffle_degradation": geometry - context,
        "geometry_signal_vs_popularity_control": geometry - popularity,
    }


def _metric(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _metric_dict(arms: dict[str, Any], arm_id: str, key: str) -> float:
    payload = arms.get(arm_id, {})
    return _metric(payload, key) if isinstance(payload, dict) else 0.0
