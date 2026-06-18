from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from hcaps.axt import AxtBatchCollator, AxtDataset
from hcaps.model import AxiomModelConfig, AxiomStructuredModel, StructuredInputAdapter
from hcaps.training.config import LossWeights
from hcaps.training.loss_registry import build_default_loss_registry

ROOT = Path(__file__).resolve().parents[2]


def test_loss_masks_prevent_missing_targets_from_contributing(p6_compiled_axt: Path) -> None:
    dataset = AxtDataset(p6_compiled_axt)
    records = [dataset[0].to_dict(), dataset[1].to_dict()]
    for record in records:
        record["loss_masks"] = deepcopy(record["loss_masks"])
        record["availability_masks"] = deepcopy(record["availability_masks"])
        record["loss_masks"]["loss_mask_relation_prediction"] = 0
        record["availability_masks"]["available_relation_targets"] = 0
    batch = AxtBatchCollator()(records)
    config = AxiomModelConfig.from_yaml(ROOT / "configs" / "model" / "structured_native_smoke.yaml")
    model = AxiomStructuredModel(config)
    model_input = StructuredInputAdapter(config)(batch)
    output = model(model_input)
    loss = build_default_loss_registry(LossWeights()).compute(
        output=output, model_input=model_input
    )

    assert loss.components["relation_prediction"].active_target_count == 0
    assert float(loss.components["relation_prediction"].loss.detach().cpu().item()) == 0.0


def test_text_projection_loss_runs_as_secondary_loss(p6_compiled_axt: Path) -> None:
    dataset = AxtDataset(p6_compiled_axt)
    batch = AxtBatchCollator()([dataset[0], dataset[1]])
    config = AxiomModelConfig.from_yaml(ROOT / "configs" / "model" / "structured_native_smoke.yaml")
    model = AxiomStructuredModel(config)
    model_input = StructuredInputAdapter(config)(batch)
    output = model(model_input)
    loss = build_default_loss_registry(LossWeights()).compute(
        output=output, model_input=model_input
    )

    assert loss.components["text_projection"].active_target_count > 0
    assert loss.components["text_projection"].weight < loss.components["structured_axc_out"].weight
