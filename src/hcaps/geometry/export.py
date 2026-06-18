"""Export gauge-invariant geometry observables to AXC-out-compatible fields."""

from __future__ import annotations

from typing import Any

from hcaps.geometry.config import GeometryMode
from hcaps.geometry.observables import GeometryObservables

FORBIDDEN_GEOMETRY_EXPORT_FRAGMENTS = (
    "truth",
    "correct",
    "is_true",
    "raw_connection",
    "connection_matrix",
    "basis_coefficient",
)


def geometry_observables_to_axc_out_fields(
    *,
    mode: GeometryMode,
    enabled: bool,
    observables: GeometryObservables,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "geometry": {
            "enabled": enabled,
            "mode": mode.value,
            "gauge_policy": "gauge_invariant_observables_only",
            "curvature_score": _scalar(observables.curvature_score),
            "holonomy_norm": _mean(observables.holonomy_norm),
            "trace_normalized": _mean(observables.trace_normalized),
            "spectrum_summary": {
                "abs_mean": _mean(observables.spectrum_abs_mean),
                "abs_max": _max(observables.spectrum_abs_max),
            },
            "path_consistency_score": _scalar(observables.path_consistency_score),
            "context_lability_score": _scalar(observables.context_lability_score),
            "loop_count": int(_scalar(observables.loop_count)),
        }
    }
    _validate_export(fields)
    return fields


def _scalar(value: object) -> float:
    if hasattr(value, "detach"):
        tensor = value.detach()
        return float(tensor.item()) if tensor.numel() == 1 else float(tensor.mean().item())
    return float(value) if isinstance(value, int | float) else 0.0


def _mean(value: object) -> float:
    if hasattr(value, "detach"):
        tensor = value.detach()
        return float(tensor.mean().item()) if tensor.numel() > 0 else 0.0
    return 0.0


def _max(value: object) -> float:
    if hasattr(value, "detach"):
        tensor = value.detach()
        return float(tensor.max().item()) if tensor.numel() > 0 else 0.0
    return 0.0


def _validate_export(payload: dict[str, Any]) -> None:
    flattened = repr(payload).lower()
    forbidden = [name for name in FORBIDDEN_GEOMETRY_EXPORT_FRAGMENTS if name in flattened]
    if forbidden:
        raise ValueError(f"forbidden geometry export field: {forbidden[0]}")
