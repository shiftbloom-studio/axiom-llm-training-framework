from __future__ import annotations

import torch

from hcaps.geometry.connection import LearnedConnection


def test_connection_matrices_are_skew_symmetric(learned_geometry_config) -> None:
    connection = LearnedConnection(learned_geometry_config)
    transition_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    output = connection(transition_ids)
    skew_error = output.connection_matrices + output.connection_matrices.transpose(-1, -2)
    assert torch.allclose(skew_error, torch.zeros_like(skew_error))
    assert output.transport_matrices.shape == (
        3,
        learned_geometry_config.fiber_dim,
        learned_geometry_config.fiber_dim,
    )


def test_matrix_exp_path_has_finite_gradients(learned_geometry_config) -> None:
    connection = LearnedConnection(learned_geometry_config)
    transition_ids = torch.tensor([0, 1], dtype=torch.long)
    output = connection(transition_ids)
    loss = output.transport_matrices.sum()
    loss.backward()
    assert connection.transition_coefficients.weight.grad is not None
    assert torch.isfinite(connection.transition_coefficients.weight.grad).all()
