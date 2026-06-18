"""Learnable context-transition connection parameterization."""

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
    """Small skew-symmetric connection basis over lateral context transitions."""

    def __init__(self, config: GeometryConfig) -> None:
        super().__init__()
        self.config = config
        self.fiber_dim = config.fiber_dim
        self.basis_count = _basis_count(config.fiber_dim, config.connection_rank)
        basis = _skew_basis(config.fiber_dim, self.basis_count)
        self.register_buffer("basis", basis)
        self.transition_coefficients = nn.Embedding(config.max_transition_types, self.basis_count)
        self.relation_coefficients = nn.Embedding(config.max_relation_types, self.basis_count)
        nn.init.zeros_(self.transition_coefficients.weight)
        nn.init.zeros_(self.relation_coefficients.weight)

    def forward(
        self,
        transition_type_ids: Tensor,
        *,
        relation_type_ids: Tensor | None = None,
    ) -> ConnectionOutput:
        if transition_type_ids.numel() == 0:
            empty_matrices = torch.zeros(
                (0, self.fiber_dim, self.fiber_dim),
                dtype=torch.float32,
                device=transition_type_ids.device,
            )
            empty_coefficients = torch.zeros(
                (0, self.basis_count),
                dtype=torch.float32,
                device=transition_type_ids.device,
            )
            return ConnectionOutput(
                connection_matrices=empty_matrices,
                transport_matrices=empty_matrices,
                coefficients=empty_coefficients,
            )
        transition_ids = torch.remainder(
            torch.abs(transition_type_ids.to(dtype=torch.long)),
            self.config.max_transition_types,
        )
        coefficients = self.transition_coefficients(transition_ids)
        if relation_type_ids is not None and relation_type_ids.numel() > 0:
            relation_ids = torch.remainder(
                torch.abs(relation_type_ids[: transition_ids.numel()].to(dtype=torch.long)),
                self.config.max_relation_types,
            )
            coefficients = coefficients + self.relation_coefficients(relation_ids)
        basis = cast(Tensor, self.basis).to(coefficients.device)
        connection = torch.einsum("eb,bxy->exy", coefficients, basis)
        connection = skew_symmetric(connection)
        transports = transport_from_connection(connection, operator=self.config.transport_operator)
        return ConnectionOutput(
            connection_matrices=connection,
            transport_matrices=transports,
            coefficients=coefficients,
        )


def skew_symmetric(matrix: Tensor) -> Tensor:
    return matrix - matrix.transpose(-1, -2)


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


def _basis_count(fiber_dim: int, connection_rank: int | None) -> int:
    full = fiber_dim * (fiber_dim - 1) // 2
    if connection_rank is None:
        return full
    return max(1, min(connection_rank, full))


def _skew_basis(fiber_dim: int, basis_count: int) -> Tensor:
    basis = torch.zeros((basis_count, fiber_dim, fiber_dim), dtype=torch.float32)
    cursor = 0
    for row in range(fiber_dim):
        for col in range(row + 1, fiber_dim):
            if cursor >= basis_count:
                return basis
            basis[cursor, row, col] = 1.0
            basis[cursor, col, row] = -1.0
            cursor += 1
    return basis
