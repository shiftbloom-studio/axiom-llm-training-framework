"""Parallel transport and path composition utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from hcaps.geometry.connection import identity_transports

MIN_CONSISTENCY_PATHS = 2


def compose_path_transport(
    edge_transports: Tensor,
    path_edge_index: Tensor,
    path_mask: Tensor,
    *,
    fiber_dim: int,
) -> Tensor:
    """Compose path transports as T_en ... T_e1 with padded edges masked out."""

    if path_edge_index.numel() == 0:
        return identity_transports(
            0,
            fiber_dim,
            device=edge_transports.device,
            dtype=edge_transports.dtype,
        )
    path_count, path_len = path_edge_index.shape
    composed = identity_transports(
        path_count,
        fiber_dim,
        device=edge_transports.device,
        dtype=edge_transports.dtype,
    )
    if edge_transports.shape[0] == 0:
        return composed
    clamped_indices = path_edge_index.clamp(min=0, max=max(0, edge_transports.shape[0] - 1))
    for step in range(path_len):
        step_transport = edge_transports.index_select(0, clamped_indices[:, step])
        active = path_mask[:, step].reshape(path_count, 1, 1).to(dtype=edge_transports.dtype)
        candidate = torch.bmm(step_transport, composed)
        composed = torch.where(active.bool(), candidate, composed)
    return composed


def transport_state(path_transport: Tensor, fiber_state: Tensor) -> Tensor:
    if path_transport.numel() == 0:
        return torch.zeros(
            (0, fiber_state.shape[-1]),
            dtype=fiber_state.dtype,
            device=fiber_state.device,
        )
    return torch.bmm(path_transport, fiber_state.unsqueeze(-1)).squeeze(-1)


def path_consistency(path_transports: Tensor) -> Tensor:
    """Return a differentiable aggregate inconsistency among available paths."""

    if path_transports.shape[0] < MIN_CONSISTENCY_PATHS:
        return torch.zeros((), dtype=path_transports.dtype, device=path_transports.device)
    flattened = path_transports.reshape(path_transports.shape[0], -1)
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    return centered.pow(2).mean()
