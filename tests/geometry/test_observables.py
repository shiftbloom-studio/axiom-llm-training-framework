from __future__ import annotations

import torch

from hcaps.geometry.config import GeometryMode
from hcaps.geometry.export import geometry_observables_to_axc_out_fields
from hcaps.geometry.observables import compute_observables


def test_identity_loop_has_zero_holonomy_norm() -> None:
    loop_transports = torch.eye(3).reshape(1, 3, 3)
    loop_mask = torch.tensor([[True, True]])
    observables = compute_observables(
        loop_transports,
        loop_mask,
        path_consistency_value=torch.tensor(0.0),
        context_lability_value=torch.tensor(0.0),
    )
    assert torch.allclose(observables.holonomy_norm, torch.tensor([0.0]))
    assert torch.allclose(observables.trace_normalized, torch.tensor([1.0]))
    assert torch.isfinite(observables.curvature_score)


def test_export_policy_is_json_safe_and_no_raw_matrices() -> None:
    loop_transports = torch.eye(2).reshape(1, 2, 2)
    observables = compute_observables(
        loop_transports,
        torch.tensor([[True]]),
        path_consistency_value=torch.tensor(0.0),
        context_lability_value=torch.tensor(0.0),
    )
    payload = geometry_observables_to_axc_out_fields(
        mode=GeometryMode.LEARNED,
        enabled=True,
        observables=observables,
    )
    flattened = repr(payload).lower()
    assert "raw" not in flattened
    assert "truth" not in flattened
    assert payload["geometry"]["trace_normalized"] == 1.0
