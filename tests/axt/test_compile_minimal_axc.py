from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle


def test_compile_minimal_axc(compiled_minimal_axc: Path) -> None:
    bundle = AxtBundle(compiled_minimal_axc)

    assert bundle.manifest.source_format == "AXC"
    assert bundle.manifest.record_count == 1
    assert "claim" in bundle.tensor_group_names()
    assert "text_projection" in bundle.tensor_group_names()
