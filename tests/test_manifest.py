from __future__ import annotations

from pathlib import Path

from hcaps.store.manifest import build_manifest_for_files, read_manifest, write_manifest
from hcaps.utils.hashing import file_sha256

FIXTURES = Path(__file__).parent / "fixtures"


def test_manifest_records_file_hashes(tmp_path: Path) -> None:
    capsule_path = FIXTURES / "valid_capsules.jsonl"
    manifest = build_manifest_for_files(
        [capsule_path],
        dataset_name="fixture-capsules",
        dataset_version="0.1.0",
    )

    assert manifest.capsule_count == 2
    assert manifest.files[0].content_hash == file_sha256(capsule_path)
    assert manifest.files[0].record_count == 2

    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, manifest)
    loaded = read_manifest(manifest_path)

    assert loaded.manifest_id == manifest.manifest_id
    assert loaded.files[0].content_hash == manifest.files[0].content_hash
