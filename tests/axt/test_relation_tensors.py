from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle


def test_relation_tensors_use_ragged_offsets(compiled_minimal_axp: Path) -> None:
    bundle = AxtBundle(compiled_minimal_axp)
    relations = bundle.read_tensor_group("relations")

    assert relations["relation_offsets"].tolist() == [0, 1, 2]
    assert relations["relation_mask"].tolist() == [1, 1]
    assert relations["relation_type_values"].shape == (2,)
