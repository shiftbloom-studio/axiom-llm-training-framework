from __future__ import annotations

from hcaps.axt import AxtBatch
from hcaps.model import AxiomModelConfig
from hcaps.model.embeddings import TypedFieldEmbeddingSet
from hcaps.model.encoder import StructuredEncoder
from hcaps.model.input_adapter import StructuredInputAdapter


def test_structured_encoder_preserves_typed_slots(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    model_input = StructuredInputAdapter(smoke_config)(axt_batch)
    slots = TypedFieldEmbeddingSet(smoke_config)(model_input)
    encoded = StructuredEncoder(smoke_config)(slots)
    assert encoded.slot_states.shape[:2] == slots.slot_tensor.shape[:2]
    assert encoded.slot_type_ids.shape == slots.slot_type_ids.shape
    assert encoded.diagnostics["slot_types_preserved"] is True
    assert "claim_state" in encoded.source_groups
    assert "text_projection" in encoded.source_groups
