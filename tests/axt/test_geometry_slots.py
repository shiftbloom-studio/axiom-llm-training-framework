from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle


def test_geometry_slots_are_present_and_do_not_store_raw_matrices(
    compiled_minimal_axp: Path,
) -> None:
    bundle = AxtBundle(compiled_minimal_axp)
    geometry = bundle.read_tensor_group("geometry_observables")

    assert "geometry_enabled" in geometry
    assert "geometry_ablation_mask" in geometry
    assert "curvature_observable_placeholder" in geometry
    assert not any("connection_matrix" in name or "raw_matrix" in name for name in geometry)
