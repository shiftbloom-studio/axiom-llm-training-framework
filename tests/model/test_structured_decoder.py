from __future__ import annotations

from hcaps.axt import AxtBatch
from hcaps.model import AxiomModelConfig, AxiomStructuredModel


def test_structured_decoder_emits_required_axc_out_heads(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    output = AxiomStructuredModel(smoke_config)(axt_batch)
    raw = output.raw_axc_out.raw_emission
    shapes = raw.head_shapes()
    for name in (
        "claim_state_logits",
        "relation_type_logits",
        "relation_target_logits",
        "provenance_source_logits",
        "evidence_span_logits",
        "epistemic_status_logits",
        "stability_logits",
        "future_summary_latent",
        "geometry_observable_values",
    ):
        assert name in shapes
        assert shapes[name][0] == 2
    assert output.raw_axc_out.validated_axc_out["invalid_forbidden_truth_field_name"] == []
    assert output.raw_axc_out.validated_axc_out["hidden_oracle_repairs"] is False
