from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle


def test_compile_minimal_axp(compiled_minimal_axp: Path) -> None:
    bundle = AxtBundle(compiled_minimal_axp)

    assert bundle.manifest.source_format == "AXP"
    assert bundle.manifest.record_count == 2
    assert len(bundle.tensor_group_names()) == 17
    assert "train_mask" in bundle.read_tensor_group("split_masks")
