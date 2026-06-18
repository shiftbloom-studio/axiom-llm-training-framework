from __future__ import annotations

import torch

from hcaps.geometry.transport import compose_path_transport


def test_path_transport_composes_in_order() -> None:
    first = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    second = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    transports = torch.stack([first, second])
    path = torch.tensor([[0, 1]], dtype=torch.long)
    mask = torch.ones_like(path, dtype=torch.bool)
    composed = compose_path_transport(transports, path, mask, fiber_dim=2)
    assert torch.allclose(composed[0], second @ first)


def test_path_transport_masks_padded_edges() -> None:
    first = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    second = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    transports = torch.stack([first, second])
    path = torch.tensor([[0, 1]], dtype=torch.long)
    mask = torch.tensor([[True, False]])
    composed = compose_path_transport(transports, path, mask, fiber_dim=2)
    assert torch.allclose(composed[0], first)


def test_empty_path_returns_empty_identity_batch() -> None:
    transports = torch.zeros((0, 2, 2))
    path = torch.zeros((0, 0), dtype=torch.long)
    mask = torch.zeros((0, 0), dtype=torch.bool)
    composed = compose_path_transport(transports, path, mask, fiber_dim=2)
    assert composed.shape == (0, 2, 2)
