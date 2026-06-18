from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle


def test_text_projection_tensors_are_present_but_secondary(compiled_minimal_axp: Path) -> None:
    bundle = AxtBundle(compiled_minimal_axp)
    text = bundle.read_tensor_group("text_projection")

    assert text["flat_text_input_ids"].shape == (2, 64)
    assert text["structured_text_input_ids"].shape == (2, 64)
    assert text["capsule_text_input_ids"].shape == (2, 64)
    assert bundle.manifest.text_projection_summary["text_projection_is_secondary"] is True
