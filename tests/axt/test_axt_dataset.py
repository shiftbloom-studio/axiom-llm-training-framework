from __future__ import annotations

from pathlib import Path

from hcaps.axt.batch import AxtBatchCollator
from hcaps.axt.dataset import AxtDataset


def test_axt_dataset_returns_structured_records(compiled_minimal_axp: Path) -> None:
    dataset = AxtDataset(compiled_minimal_axp)
    record = dataset[0]

    assert len(dataset) == 2
    assert "input_ids" not in record.to_dict()
    assert "flat_text_input_ids" in record.text_projection
    assert "loss_mask_text_projection" in record.loss_masks


def test_axt_batch_collator_preserves_structured_groups(compiled_minimal_axp: Path) -> None:
    dataset = AxtDataset(compiled_minimal_axp)
    batch = AxtBatchCollator()([dataset[0], dataset[1]])

    assert batch.size == 2
    assert "text_projection" in batch.to_dict()["groups"]
    assert "negative_samples" in batch.to_dict()["groups"]
