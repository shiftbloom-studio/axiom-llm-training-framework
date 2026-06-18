"""Unweighted geometry regularizer terms for P6."""

from __future__ import annotations

import torch
from torch import Tensor

from hcaps.geometry.observables import GeometryObservables

MIN_SMOOTHNESS_NODES = 2


def geometry_regularizers(
    *,
    connection_matrices: Tensor,
    transport_matrices: Tensor,
    path_consistency_value: Tensor,
    conditioning_features: Tensor,
    observables: GeometryObservables,
) -> dict[str, Tensor]:
    zero = torch.zeros((), dtype=conditioning_features.dtype, device=conditioning_features.device)
    connection_norm = connection_matrices.pow(2).mean() if connection_matrices.numel() > 0 else zero
    identity_bias = zero
    if transport_matrices.numel() > 0:
        identity = torch.eye(
            transport_matrices.shape[-1],
            dtype=transport_matrices.dtype,
            device=transport_matrices.device,
        ).expand_as(transport_matrices)
        identity_bias = (transport_matrices - identity).pow(2).mean()
    context_smoothness = _context_smoothness(conditioning_features)
    usage = -conditioning_features.pow(2).mean().sqrt()
    return {
        "connection_norm": connection_norm,
        "curvature_energy": observables.loop_energy,
        "path_consistency": path_consistency_value,
        "context_smoothness": context_smoothness,
        "transport_identity_bias": identity_bias,
        "non_degenerate_usage": usage,
    }


def zero_regularizers(
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    zero = torch.zeros((), dtype=dtype, device=device)
    return {
        "connection_norm": zero,
        "curvature_energy": zero,
        "path_consistency": zero,
        "context_smoothness": zero,
        "transport_identity_bias": zero,
        "non_degenerate_usage": zero,
    }


def _context_smoothness(conditioning_features: Tensor) -> Tensor:
    if conditioning_features.shape[0] < MIN_SMOOTHNESS_NODES:
        return torch.zeros(
            (),
            dtype=conditioning_features.dtype,
            device=conditioning_features.device,
        )
    return (conditioning_features[1:] - conditioning_features[:-1]).pow(2).mean()
