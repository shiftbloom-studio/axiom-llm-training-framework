from __future__ import annotations

from pathlib import Path

import orjson

from hcaps.format.package import inspect_package, validate_package


def test_canonical_minimal_axp_package_validates(axf_examples_dir: Path) -> None:
    package_path = axf_examples_dir / "minimal_dataset.axp"

    report = validate_package(package_path)
    details = inspect_package(package_path)

    assert report.ok
    assert details["dataset_name"] == "Minimal AXF v0.1 Conformance Dataset"
    assert details["capsule_count"] == 2
    assert details["file_count"] == 8


def test_minimal_tensor_bundle_fixture_declares_no_tensors(axf_examples_dir: Path) -> None:
    manifest = orjson.loads((axf_examples_dir / "minimal_tensor_bundle.axt").read_bytes())

    assert manifest["format"] == "AXT"
    assert manifest["format_version"] == "0.1.0"
    assert manifest["contains_tensors"] is False
    assert manifest["tensor_compilation"] == "deferred_to_step_3"
