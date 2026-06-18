import json
from pathlib import Path

from hcaps.training_bridge.dataset import AxcTrainingDataset
from hcaps.training_bridge.tokenizer import WhitespaceTokenizer


def test_axc_training_dataset_flat_mode(tmp_path: Path) -> None:
    # Create a small AXC file with two capsules
    axc_path = tmp_path / "sample.axc"
    capsules = [
        {
            "ids": {"capsule_id": "c1", "claim_family_id": "f1"},
            "claim": {"canonical": "The sky is blue"},
            "surface_forms": {"normalized_summary": "The sky is blue"},
            "epistemic_state": {},
        },
        {
            "ids": {"capsule_id": "c2", "claim_family_id": "f2"},
            "claim": {"canonical": "Grass is green"},
            "surface_forms": {"normalized_summary": "Grass is green"},
            "epistemic_state": {},
        },
    ]
    with open(axc_path, "w", encoding="utf-8") as f:
        for cap in capsules:
            f.write(json.dumps(cap) + "\n")
    tok = WhitespaceTokenizer()
    tok.fit(["The sky is blue", "Grass is green"])
    ds = AxcTrainingDataset(axc_path, tok, mode="flat_text")
    assert len(ds) == 2
    item0 = ds[0]
    assert item0["capsule_id"] == "c1"
    assert item0["claim_family_id"] == "f1"
    assert len(item0["input_ids"]) == 4
    assert item0["labels"] == item0["input_ids"]
