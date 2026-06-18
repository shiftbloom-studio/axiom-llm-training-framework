from __future__ import annotations

from pathlib import Path

from hcaps.format.package import build_package_from_axc_stream, validate_package
from hcaps.format.streams import read_axc_stream


def test_minimal_axp_package_can_be_created_from_capsules(
    axf_examples_dir: Path,
    tmp_path: Path,
) -> None:
    capsules = list(read_axc_stream(axf_examples_dir / "minimal_capsules.axc"))
    package_path = tmp_path / "created.axp"

    manifest = build_package_from_axc_stream(
        package_path,
        dataset_name="Created AXF Fixture",
        capsules=capsules,
    )
    report = validate_package(package_path)

    assert manifest.counts.capsules == 1
    assert report.ok
    assert (package_path / "axiom.json").exists()
    assert (package_path / "data" / "capsules.axc").exists()
    assert (package_path / "data" / "sources.axsrc").exists()
    assert (package_path / "data" / "relations.axr").exists()
    assert not (package_path / "data" / "capsules.axc.jsonl").exists()
    assert (package_path / "splits" / "train.json").exists()
    assert (package_path / "splits" / "validation.json").exists()
    assert (package_path / "splits" / "temporal_holdout.json").exists()
    assert (package_path / "manifests" / "files.json").exists()
    assert (package_path / "manifests" / "hashes.json").exists()
    assert (package_path / "manifests" / "construction_report.json").exists()
    assert (package_path / "manifests" / "leakage_report.json").exists()
