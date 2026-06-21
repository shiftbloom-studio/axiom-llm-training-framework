"""Open, time-aware, non-orthogonal context-transport connection.

Per the highest-order HKR paradigm (docs/concept/HKR_HIGHEST_ORDER_PARADIGM.md):
context is open-dimensional (not an enumerated type table), the fact-core is not
invariant, and reality drifts over time. The transport here is therefore generated
by a learned function of *continuous* per-edge context features (the endpoint fiber
states plus their temporal features, assembled by the module) and is a GENERAL
invertible map -- matrix_exp of a non-skew generator -- so it can MOVE the core,
not an SO(n) rotation that merely reframes an invariant core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from hcaps.geometry.config import GeometryConfig, TransportOperator


@dataclass(frozen=True)
class ConnectionOutput:
    connection_matrices: Tensor
    transport_matrices: Tensor
    coefficients: Tensor


class LearnedConnection(nn.Module):
    """Generate per-edge transports from continuous, open context features.

    Input ``edge_context`` is ``[num_edges, context_in_dim]`` (the module builds it
    from the endpoint fiber states and their temporal features). There are no
    enumerated transition/relation type tables, so context is not a closed list.
    """

    def __init__(self, config: GeometryConfig, *, context_in_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.fiber_dim = config.fiber_dim
        # endpoint fibers (2 * fiber_dim) + their two temporal scalars
        self.context_in_dim = context_in_dim or (2 * config.fiber_dim + 2)
        hidden = (
            config.connection_hidden
            if config.connection_hidden > 0
            else max(32, 2 * config.fiber_dim)
        )
        self.generator = nn.Sequential(
            nn.Linear(self.context_in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.fiber_dim * config.fiber_dim),
        )
        # Small, non-zero init: transports start near (not exactly) identity -- no
        # privileged flat/invariant prior -- while staying bounded for matrix_exp.
        final = cast(nn.Linear, self.generator[-1])
        nn.init.normal_(final.weight, std=1e-3)
        nn.init.zeros_(final.bias)
        self.scale = float(config.connection_scale)

    def forward(self, edge_context: Tensor) -> ConnectionOutput:
        n = self.fiber_dim
        device = edge_context.device
        if edge_context.numel() == 0:
            empty = torch.zeros((0, n, n), dtype=torch.float32, device=device)
            return ConnectionOutput(
                connection_matrices=empty,
                transport_matrices=empty,
                coefficients=torch.zeros((0, n * n), dtype=torch.float32, device=device),
            )
        edges = edge_context.shape[0]
        generators = self.generator(edge_context.float()).reshape(edges, n, n) * self.scale
        generators = _bound_norm(generators, self.config.connection_max_norm)
        transports = transport_from_connection(
            generators, operator=self.config.transport_operator
        )
        return ConnectionOutput(
            connection_matrices=generators,
            transport_matrices=transports,
            coefficients=generators.reshape(edges, -1),
        )


def _bound_norm(generators: Tensor, max_norm: float) -> Tensor:
    """Soft-cap the per-edge Frobenius norm so matrix_exp stays numerically bounded."""
    if max_norm <= 0:
        return generators
    edges = generators.shape[0]
    norm = generators.reshape(edges, -1).norm(dim=-1).clamp_min(1e-6)
    factor = (max_norm / norm).clamp_max(1.0).reshape(edges, 1, 1)
    return generators * factor


def transport_from_connection(connection: Tensor, *, operator: TransportOperator) -> Tensor:
    if connection.numel() == 0:
        return connection
    if operator == TransportOperator.CAYLEY:
        identity = torch.eye(connection.shape[-1], dtype=connection.dtype, device=connection.device)
        identity = identity.expand(connection.shape[0], -1, -1)
        return cast(
            Tensor,
            torch.linalg.solve(identity - 0.5 * connection, identity + 0.5 * connection),
        )
    return cast(Tensor, torch.linalg.matrix_exp(connection))


def identity_transports(
    count: int,
    fiber_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    identity = torch.eye(fiber_dim, dtype=dtype, device=device)
    return identity.expand(count, fiber_dim, fiber_dim).clone()
