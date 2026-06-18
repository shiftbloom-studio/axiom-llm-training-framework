from __future__ import annotations

import torch

from hcaps.axt import AxtBatch
from hcaps.model import AxiomModelConfig
from hcaps.model.factory import AxiomStructuredModel
from tests.model._helpers import config_with


def test_epistemic_router_returns_finite_diagnostics(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    model = AxiomStructuredModel(smoke_config)
    output = model(axt_batch)
    assert output.router_diagnostics["enabled"] is True
    assert output.router_diagnostics["forbidden_truth_outputs"] is False
    assert torch.isfinite(output.auxiliary_logits["head_weights"]).all()


def test_router_off_path_works_without_truth_outputs(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    config = config_with(smoke_config, ablations={"router_off": True})
    output = AxiomStructuredModel(config)(axt_batch)
    assert output.router_diagnostics["enabled"] is False
    assert output.router_diagnostics["forbidden_truth_outputs"] is False
