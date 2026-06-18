from __future__ import annotations

from hcaps.axt import AxtBatch
from hcaps.model import AxiomModelConfig, AxiomStructuredModel
from tests.model._helpers import config_with


def test_text_projection_emits_batch_sequence_vocab_logits(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    output = AxiomStructuredModel(smoke_config)(axt_batch)
    assert output.text_projection_logits is not None
    assert output.text_projection_logits.shape == (2, 32, smoke_config.text_vocab_size)
    assert output.raw_axc_out.text_projection["role"] == (
        "secondary_projection_and_comparison_interface"
    )


def test_structure_only_ablation_disables_text_projection_head(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    config = config_with(smoke_config, ablations={"structure_only_no_text_projection": True})
    output = AxiomStructuredModel(config)(axt_batch)
    assert output.text_projection_logits is None
    assert output.raw_axc_out.text_projection["available"] is False
