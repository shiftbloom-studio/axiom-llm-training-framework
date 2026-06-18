"""Geometry slot tensor compilation for AXT."""

from __future__ import annotations

from typing import Any

from .schema import MISSING_FLOAT, MISSING_INT, TensorGroup, as_f32, as_i64, stable_int64


def compile_geometry_slots(
    capsules: list[Any],
    *,
    include_geometry_slots: bool,
) -> tuple[TensorGroup, dict[str, int]]:
    """Reserve gauge-invariant geometry slots without semantic raw matrices."""

    enabled: list[int] = []
    context_node_ids: list[int] = []
    context_edge_ids: list[int] = []
    context_loop_ids: list[int] = []
    transport_path_ids: list[int] = []
    curvature: list[float] = []
    holonomy: list[float] = []
    trace: list[float] = []
    spectrum: list[float] = []
    context_lability: list[float] = []
    loss_mask: list[int] = []
    ablation_mask: list[int] = []

    active = 0
    for capsule in capsules:
        geometry = capsule.geometry
        is_enabled = bool(include_geometry_slots and geometry.enabled)
        if is_enabled:
            active += 1
        enabled.append(int(is_enabled))
        context_node_ids.append(stable_int64("context", capsule.ids.context_id))
        context_edge_ids.append(MISSING_INT)
        context_loop_ids.append(
            stable_int64("geometry_loop", *geometry.loop_identifiers)
            if geometry.loop_identifiers
            else MISSING_INT
        )
        transport_path_ids.append(
            stable_int64("transport_path", *geometry.context_sequence)
            if geometry.context_sequence
            else MISSING_INT
        )
        curvature.append(
            float(geometry.curvature_score)
            if is_enabled and geometry.curvature_score is not None
            else MISSING_FLOAT
        )
        holonomy.append(float(geometry.transport_observables.get("holonomy_norm", MISSING_FLOAT)))
        trace.append(float(geometry.transport_observables.get("trace_summary", MISSING_FLOAT)))
        spectrum.append(
            float(geometry.transport_observables.get("spectrum_summary", MISSING_FLOAT))
        )
        context_lability.append(
            float(geometry.transport_observables.get("context_lability", MISSING_FLOAT))
        )
        loss_mask.append(int(is_enabled and geometry.curvature_score is not None))
        ablation_mask.append(1)

    group: TensorGroup = {
        "geometry_enabled": as_i64(enabled),
        "context_node_ids": as_i64(context_node_ids),
        "context_edge_ids": as_i64(context_edge_ids),
        "context_loop_ids": as_i64(context_loop_ids),
        "transport_path_ids": as_i64(transport_path_ids),
        "curvature_observable_placeholder": as_f32(curvature),
        "curvature_score": as_f32(curvature),
        "holonomy_norm_placeholder": as_f32(holonomy),
        "holonomy_norm": as_f32(holonomy),
        "trace_summary_placeholder": as_f32(trace),
        "trace_summary": as_f32(trace),
        "spectrum_summary_placeholder": as_f32(spectrum),
        "spectrum_summary": as_f32(spectrum),
        "context_lability": as_f32(context_lability),
        "geometry_loss_mask": as_i64(loss_mask),
        "geometry_ablation_mask": as_i64(ablation_mask),
    }
    return group, {"geometry_enabled_records": active, "geometry_slots_reserved": len(capsules)}
