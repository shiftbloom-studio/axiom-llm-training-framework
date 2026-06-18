from __future__ import annotations

from hcaps.axt import AxtBatch
from hcaps.model import AxiomModelConfig
from hcaps.model.input_adapter import StructuredInputAdapter


def test_input_adapter_consumes_p3_runtime_batch(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    model_input = StructuredInputAdapter(smoke_config)(axt_batch)
    assert model_input.batch_size == 2
    assert "claim" in model_input.groups
    assert "relation_neighborhoods" in model_input.groups
    assert "epistemic_state" in model_input.groups
    assert "available_relation_targets" in model_input.availability_masks
    assert "loss_mask_text_projection" in model_input.loss_masks


def test_input_adapter_keeps_temporal_and_lateral_context_separate(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    model_input = StructuredInputAdapter(smoke_config)(axt_batch)
    assert model_input.group("temporal")
    assert model_input.group("lateral_context")
    assert set(model_input.group("temporal")).isdisjoint(model_input.group("lateral_context"))
