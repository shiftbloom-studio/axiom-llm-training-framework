from __future__ import annotations

from hcaps.axt import AxtBatch
from hcaps.model import AxiomModelConfig, AxiomStructuredModel
from tests.model._helpers import config_with


def test_text_only_ablation_keeps_text_projection_and_native_decoder_surface(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    config = config_with(smoke_config, ablations={"text_only": True})
    output = AxiomStructuredModel(config)(axt_batch)
    assert output.text_projection_logits is not None
    assert output.raw_axc_out.raw_emission.claim_state_logits.shape[0] == 2
    assert output.diagnostics["slot_count_by_type"] == {"text_projection": 1}


def test_side_channel_ablations_are_explicit_in_diagnostics(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    config = config_with(
        smoke_config,
        ablations={
            "no_provenance": True,
            "no_context": True,
            "no_provider_context": True,
            "no_side_channels": True,
            "relation_neighborhood_off": True,
        },
    )
    output = AxiomStructuredModel(config)(axt_batch)
    ablations = output.diagnostics["active_ablation_modes"]
    assert ablations["no_side_channels"] is True
    assert output.diagnostics["stage_diagnostics"]["provenance_conditioning"]["enabled"] is False
