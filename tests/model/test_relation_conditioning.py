from __future__ import annotations

from hcaps.axt import AxtBatch
from hcaps.model import AxiomModelConfig
from hcaps.model.embeddings import TypedFieldEmbeddingSet
from hcaps.model.encoder import StructuredEncoder
from hcaps.model.input_adapter import StructuredInputAdapter
from hcaps.model.relation_conditioning import RelationNeighborhoodConditioner
from tests.model._helpers import config_with


def test_relation_conditioning_keeps_hypergraph_path_open(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    model_input = StructuredInputAdapter(smoke_config)(axt_batch)
    slots = TypedFieldEmbeddingSet(smoke_config)(model_input)
    encoded = StructuredEncoder(smoke_config)(slots)
    output = RelationNeighborhoodConditioner(smoke_config)(
        encoded.slot_states,
        encoded.slot_mask,
        model_input,
    )
    assert output.slot_states.shape == encoded.slot_states.shape
    assert output.diagnostics["hypergraph"]["hardcoded_binary_only"] is False
    assert output.diagnostics["hypergraph"]["supports_future_n_ary_tensors"] is True


def test_relation_conditioning_ablation_is_shape_compatible(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    config = config_with(smoke_config, ablations={"no_relations": True})
    model_input = StructuredInputAdapter(config)(axt_batch)
    slots = TypedFieldEmbeddingSet(config)(model_input)
    encoded = StructuredEncoder(config)(slots)
    output = RelationNeighborhoodConditioner(config)(
        encoded.slot_states,
        encoded.slot_mask,
        model_input,
    )
    assert output.slot_states.shape == encoded.slot_states.shape
    assert output.diagnostics["enabled"] is False
