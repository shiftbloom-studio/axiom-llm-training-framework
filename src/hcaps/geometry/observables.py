"""Gauge-invariant holonomy and curvature observables."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GeometryObservables:
    holonomy_norm: Tensor
    trace_normalized: Tensor
    spectrum_abs_mean: Tensor
    spectrum_abs_max: Tensor
    loop_energy: Tensor
    curvature_score: Tensor
    path_consistency_score: Tensor
    context_lability_score: Tensor
    loop_count: Tensor

    def tensor_summary(self) -> dict[str, Tensor]:
        return {
            "holonomy_norm": self.holonomy_norm,
            "trace_normalized": self.trace_normalized,
            "spectrum_abs_mean": self.spectrum_abs_mean,
            "spectrum_abs_max": self.spectrum_abs_max,
            "loop_energy": self.loop_energy,
            "curvature_score": self.curvature_score,
            "path_consistency_score": self.path_consistency_score,
            "context_lability_score": self.context_lability_score,
            "loop_count": self.loop_count,
        }


def compute_observables(
    loop_transports: Tensor,
    loop_mask: Tensor,
    *,
    path_consistency_value: Tensor,
    context_lability_value: Tensor,
) -> GeometryObservables:
    device = loop_transports.device
    dtype = loop_transports.dtype
    if loop_transports.numel() == 0:
        zero = torch.zeros((), dtype=dtype, device=device)
        return GeometryObservables(
            holonomy_norm=torch.zeros((0,), dtype=dtype, device=device),
            trace_normalized=torch.zeros((0,), dtype=dtype, device=device),
            spectrum_abs_mean=torch.zeros((0,), dtype=dtype, device=device),
            spectrum_abs_max=torch.zeros((0,), dtype=dtype, device=device),
            loop_energy=zero,
            curvature_score=zero,
            path_consistency_score=path_consistency_value,
            context_lability_score=context_lability_value,
            loop_count=zero,
        )
    fiber_dim = loop_transports.shape[-1]
    identity = torch.eye(fiber_dim, dtype=dtype, device=device).expand_as(loop_transports)
    delta = loop_transports - identity
    holonomy_norm = torch.linalg.matrix_norm(delta, ord="fro", dim=(-2, -1))
    trace_normalized = torch.diagonal(loop_transports, dim1=-2, dim2=-1).sum(dim=-1) / fiber_dim
    eigenvalues = torch.linalg.eigvals(loop_transports)
    spectrum_abs = torch.abs(eigenvalues)
    spectrum_abs_mean = spectrum_abs.mean(dim=-1).real
    spectrum_abs_max = spectrum_abs.max(dim=-1).values.real
    active_lengths = loop_mask.sum(dim=-1).clamp_min(1).to(dtype=dtype)
    loop_energy = holonomy_norm.pow(2).mean()
    curvature_score = (holonomy_norm / active_lengths).mean()
    return GeometryObservables(
        holonomy_norm=holonomy_norm,
        trace_normalized=trace_normalized,
        spectrum_abs_mean=spectrum_abs_mean,
        spectrum_abs_max=spectrum_abs_max,
        loop_energy=loop_energy,
        curvature_score=curvature_score,
        path_consistency_score=path_consistency_value,
        context_lability_score=context_lability_value,
        loop_count=torch.tensor(float(loop_transports.shape[0]), dtype=dtype, device=device),
    )


def empty_observables(
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> GeometryObservables:
    zero = torch.zeros((), dtype=dtype, device=device)
    return GeometryObservables(
        holonomy_norm=torch.zeros((0,), dtype=dtype, device=device),
        trace_normalized=torch.zeros((0,), dtype=dtype, device=device),
        spectrum_abs_mean=torch.zeros((0,), dtype=dtype, device=device),
        spectrum_abs_max=torch.zeros((0,), dtype=dtype, device=device),
        loop_energy=zero,
        curvature_score=zero,
        path_consistency_score=zero,
        context_lability_score=zero,
        loop_count=zero,
    )
