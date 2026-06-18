from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import orjson

from hcaps.format.hashing import canonical_json_bytes, file_hash, record_hash, semantic_content_hash
from hcaps.format.streams import read_axc_stream, write_axc_stream


def test_axc_record_hash_is_stable_across_reload(axf_examples_dir: Path, tmp_path: Path) -> None:
    capsule = next(read_axc_stream(axf_examples_dir / "minimal_capsules.axc"))
    record = capsule.model_dump(mode="json")
    canonical_record = orjson.loads(canonical_json_bytes(record))

    output = tmp_path / "capsules.axc"
    write_axc_stream(output, [capsule])
    reloaded = next(read_axc_stream(output)).model_dump(mode="json")

    assert record_hash(record) == record_hash(reloaded)
    assert semantic_content_hash(record) == semantic_content_hash(reloaded)
    assert record_hash(record) == record_hash(canonical_record)


def test_axc_hash_changes_when_semantic_field_changes(axf_examples_dir: Path) -> None:
    capsule = next(read_axc_stream(axf_examples_dir / "minimal_capsules.axc"))
    record = capsule.model_dump(mode="json")
    changed = deepcopy(record)
    changed["claim"]["canonical_text"] = "A covered container changes this fixture claim."

    assert record_hash(record) != record_hash(changed)
    assert semantic_content_hash(record) != semantic_content_hash(changed)


def test_axp_package_hash_manifest_matches_files(axf_examples_dir: Path) -> None:
    package_path = axf_examples_dir / "minimal_dataset.axp"
    hashes = orjson.loads((package_path / "manifests" / "hashes.json").read_bytes())

    assert hashes["data/capsules.axc"] == file_hash(package_path / "data" / "capsules.axc")
