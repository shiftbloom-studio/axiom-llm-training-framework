from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle


def test_epistemic_tensors_include_values_confidence_and_proxy_metadata(
    compiled_minimal_axp: Path,
) -> None:
    bundle = AxtBundle(compiled_minimal_axp)
    epistemic = bundle.read_tensor_group("epistemic_state")

    assert epistemic["ontology_compatibility_value"].shape == (2,)
    assert epistemic["uncertainty_confidence"].shape == (2,)
    assert epistemic["proxy_method_idx"].shape == (2, 5)
    assert epistemic["proxy_basis_hash"].shape == (2, 5)
