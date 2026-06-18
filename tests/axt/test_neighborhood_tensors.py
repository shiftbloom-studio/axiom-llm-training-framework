from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle


def test_neighborhood_tensors_keep_hypergraph_path_open(compiled_minimal_axp: Path) -> None:
    bundle = AxtBundle(compiled_minimal_axp)
    neighborhoods = bundle.read_tensor_group("relation_neighborhoods")

    assert neighborhoods["neighborhood_masks"].tolist() == [1, 1]
    assert neighborhoods["hypergraph_reserved"].tolist() == [1]
    assert "hyperedge_incidence_values" in neighborhoods
