from __future__ import annotations

import torch

from hcaps.geometry.connection import LearnedConnection


def test_transports_are_invertible_and_finite(learned_geometry_config) -> None:
    conn = LearnedConnection(learned_geometry_config)
    n = learned_geometry_config.fiber_dim
    edge_context = torch.randn(2, conn.context_in_dim)
    output = conn(edge_context)
    assert output.transport_matrices.shape == (2, n, n)
    assert torch.isfinite(output.transport_matrices).all()
    # matrix_exp of a real generator is always invertible (det = exp(trace) > 0); unlike an
    # SO(n) rotation it may be non-orthogonal, so context can MOVE the core, not only reframe it.
    dets = torch.linalg.det(output.transport_matrices)
    assert torch.all(dets.abs() > 0)


def test_connection_gradients_flow(learned_geometry_config) -> None:
    conn = LearnedConnection(learned_geometry_config)
    edge_context = torch.randn(2, conn.context_in_dim, requires_grad=True)
    output = conn(edge_context)
    output.transport_matrices.sum().backward()
    grads = [p.grad for p in conn.parameters() if p.grad is not None]
    assert grads, "no connection parameter received gradient"
    assert all(torch.isfinite(g).all() for g in grads)
