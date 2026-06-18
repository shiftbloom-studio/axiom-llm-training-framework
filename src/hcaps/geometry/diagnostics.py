"""JSON-safe geometry diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from hcaps.geometry.batching import ClaimFieldGraphBatch
from hcaps.geometry.config import GeometryMode
from hcaps.geometry.connection import ConnectionOutput
from hcaps.geometry.observables import GeometryObservables


@dataclass(frozen=True)
class GeometryDiagnostics:
    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


def build_geometry_diagnostics(
    *,
    mode: GeometryMode,
    graph_batch: ClaimFieldGraphBatch,
    connection: ConnectionOutput | None,
    observables: GeometryObservables,
    no_loops: bool,
    empty_graph: bool,
) -> GeometryDiagnostics:
    holonomy = observables.holonomy_norm.detach()
    trace = observables.trace_normalized.detach()
    connection_norms = (
        torch.linalg.matrix_norm(connection.connection_matrices.detach(), dim=(-2, -1))
        if connection is not None and connection.connection_matrices.numel() > 0
        else torch.zeros((0,), dtype=torch.float32, device=graph_batch.device)
    )
    loop_lengths = graph_batch.loop_mask.sum(dim=-1).to(dtype=torch.float32)
    values = {
        "num_nodes": graph_batch.num_nodes,
        "num_edges": graph_batch.num_edges,
        "num_context_edges": graph_batch.num_context_edges,
        "num_paths": int(graph_batch.path_edge_index.shape[0]),
        "num_loops": int(graph_batch.loop_path_index.shape[0]),
        "mean_loop_length": _mean(loop_lengths),
        "mean_holonomy_norm": _mean(holonomy),
        "max_holonomy_norm": _max(holonomy),
        "mean_trace_normalized": _mean(trace),
        "curvature_nonzero_fraction": _nonzero_fraction(holonomy),
        "connection_norm_mean": _mean(connection_norms),
        "connection_norm_max": _max(connection_norms),
        "no_loops": no_loops,
        "empty_graph": empty_graph,
        "geometry_mode": mode.value,
        "gauge_invariant_only": True,
        "raw_gauge_matrices_emitted": False,
    }
    return GeometryDiagnostics(values=values)


def _mean(value: Tensor) -> float:
    return float(value.mean().item()) if value.numel() > 0 else 0.0


def _max(value: Tensor) -> float:
    return float(value.max().item()) if value.numel() > 0 else 0.0


def _nonzero_fraction(value: Tensor) -> float:
    return float(value.ne(0).to(dtype=torch.float32).mean().item()) if value.numel() > 0 else 0.0
