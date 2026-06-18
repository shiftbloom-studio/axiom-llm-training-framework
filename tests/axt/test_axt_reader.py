from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle


def test_axt_reader_loads_selected_groups_and_source_index(compiled_minimal_axp: Path) -> None:
    bundle = AxtBundle(compiled_minimal_axp)

    ids = bundle.read_tensor_group("ids")
    source_index = bundle.source_index()

    assert ids["capsule_index"].tolist() == [0, 1]
    assert source_index["source_format"] == "AXP"
    assert len(source_index["records"]) == 2
