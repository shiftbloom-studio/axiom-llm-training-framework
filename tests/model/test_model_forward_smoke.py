from __future__ import annotations

import torch

from hcaps.axt import AxtBatch
from hcaps.model import AxiomModelConfig, AxiomStructuredModel


def test_model_forward_smoke_on_p3_axt_fixture(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    model = AxiomStructuredModel(smoke_config)
    output = model(axt_batch)
    assert output.latent_state.shape == (2, smoke_config.model_dim)
    assert output.raw_axc_out is not None
    assert output.text_projection_logits is not None
    assert output.router_diagnostics
    assert output.geometry_diagnostics
    assert torch.isfinite(output.latent_state).all()
    assert torch.isfinite(output.slot_states).all()
    assert output.diagnostics["slot_count_by_type"]["relations"] == 1
