from __future__ import annotations

from pathlib import Path
from typing import Any

from hcaps.format.capsule import AxcCapsule
from hcaps.format.package import (
    build_package_from_axc_stream,
    create_package_skeleton,
    validate_package,
)


def test_axp_package_skeleton_can_be_created_and_validated(tmp_path: Path) -> None:
    package_path = tmp_path / "fixture.axp"

    create_package_skeleton(package_path, dataset_name="Fixture")
    report = validate_package(package_path)

    assert report.ok
    assert (package_path / "compiled" / "README.md").exists()


def test_axp_file_hash_mismatch_is_detected(minimal_axc_record: Any, tmp_path: Path) -> None:
    package_path = tmp_path / "fixture.axp"
    capsule = AxcCapsule.model_validate(minimal_axc_record)
    build_package_from_axc_stream(package_path, dataset_name="Fixture", capsules=[capsule])
    (package_path / "data" / "capsules.axc").write_text("corrupted\n", encoding="utf-8")

    report = validate_package(package_path)

    assert not report.ok
    assert report.issues[0].code == "file_hash_mismatch"
