from __future__ import annotations

import numpy as np
import torch

from hcaps.geometry.config import TransportOperator
from hcaps.geometry.connection import transport_from_connection
from hcaps.geometry.reference import matrix_exp_reference, observables_reference


def test_reference_matrix_exp_matches_torch_on_tiny_skew_matrix() -> None:
    matrix = torch.tensor([[0.0, 0.2], [-0.2, 0.0]])
    torch_transport = transport_from_connection(
        matrix.reshape(1, 2, 2),
        operator=TransportOperator.MATRIX_EXP,
    )[0]
    reference_transport = matrix_exp_reference(matrix.numpy())
    assert np.allclose(torch_transport.detach().numpy(), reference_transport, atol=1e-5)


def test_reference_observables_identity() -> None:
    payload = observables_reference(np.eye(3))
    assert payload["holonomy_norm"] == 0.0
    assert payload["trace_normalized"] == 1.0
