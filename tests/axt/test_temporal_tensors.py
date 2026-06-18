from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle


def test_temporal_tensors_are_separate_from_lateral_context(compiled_minimal_axp: Path) -> None:
    bundle = AxtBundle(compiled_minimal_axp)
    temporal = bundle.read_tensor_group("temporal")
    lateral = bundle.read_tensor_group("lateral_context")

    assert "valid_as_of_timestamp" in temporal
    assert "temporal_cutoff_mask" in temporal
    assert "domain_idx" in lateral
    assert "valid_as_of_timestamp" not in lateral
    assert temporal["temporal_cutoff_mask"].tolist() == [1, 1]
