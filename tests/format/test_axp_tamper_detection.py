from __future__ import annotations

from pathlib import Path
from shutil import copytree

import orjson

from hcaps.format.package import validate_package


def _copy_example_package(axf_examples_dir: Path, tmp_path: Path) -> Path:
    source = axf_examples_dir / "minimal_dataset.axp"
    target = tmp_path / "tampered.axp"
    copytree(source, target)
    return target


def _issue_codes(package_path: Path) -> set[str]:
    return {issue.code for issue in validate_package(package_path).issues}


def test_axp_validation_fails_when_required_file_is_missing(
    axf_examples_dir: Path,
    tmp_path: Path,
) -> None:
    package_path = _copy_example_package(axf_examples_dir, tmp_path)
    (package_path / "manifests" / "leakage_report.json").unlink()

    assert "missing_required_file" in _issue_codes(package_path)


def test_axp_validation_fails_when_file_hash_changes(
    axf_examples_dir: Path,
    tmp_path: Path,
) -> None:
    package_path = _copy_example_package(axf_examples_dir, tmp_path)
    source_path = package_path / "data" / "sources.axsrc"
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace("Synthetic", "Altered", 1),
        encoding="utf-8",
    )

    codes = _issue_codes(package_path)

    assert "file_hash_mismatch" in codes
    assert "hash_manifest_mismatch" in codes


def test_axp_validation_fails_when_split_references_missing_capsule(
    axf_examples_dir: Path,
    tmp_path: Path,
) -> None:
    package_path = _copy_example_package(axf_examples_dir, tmp_path)
    (package_path / "splits" / "train.json").write_bytes(
        orjson.dumps({"capsule_ids": ["axc:sha256:" + ("a" * 64)]}) + b"\n"
    )

    assert "missing_capsule_reference" in _issue_codes(package_path)


def test_axp_validation_fails_when_relation_references_missing_target(
    axf_examples_dir: Path,
    tmp_path: Path,
) -> None:
    package_path = _copy_example_package(axf_examples_dir, tmp_path)
    relation_path = package_path / "data" / "relations.axr"
    records = [
        orjson.loads(line) for line in relation_path.read_bytes().splitlines() if line.strip()
    ]
    records[0]["target_claim_family_id"] = "claimfam:missing:" + ("a" * 24)
    records[0]["target_capsule_id"] = "axc:sha256:" + ("b" * 64)
    relation_path.write_bytes(
        b"".join(orjson.dumps(record, option=orjson.OPT_SORT_KEYS) + b"\n" for record in records)
    )

    assert "missing_relation_target" in _issue_codes(package_path)


def test_axp_validation_fails_when_source_span_references_missing_source(
    axf_examples_dir: Path,
    tmp_path: Path,
) -> None:
    package_path = _copy_example_package(axf_examples_dir, tmp_path)
    capsules_path = package_path / "data" / "capsules.axc"
    records = [
        orjson.loads(line) for line in capsules_path.read_bytes().splitlines() if line.strip()
    ]
    records[0]["surface_forms"]["source_spans"][0]["source_id"] = "src:fixture:" + ("a" * 24)
    capsules_path.write_bytes(
        b"".join(orjson.dumps(record, option=orjson.OPT_SORT_KEYS) + b"\n" for record in records)
    )

    assert "axc_schema_validation_failed" in _issue_codes(package_path)
