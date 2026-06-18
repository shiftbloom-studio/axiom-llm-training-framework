from __future__ import annotations

from typing import Any

from hcaps.format.hashing import record_hash, semantic_content_hash


def test_hashes_are_stable_under_key_reordering(minimal_axc_record: Any) -> None:
    reordered = dict(reversed(list(minimal_axc_record.items())))

    assert record_hash(minimal_axc_record) == record_hash(reordered)
    assert semantic_content_hash(minimal_axc_record) == semantic_content_hash(reordered)


def test_semantic_hash_excludes_generated_ids(minimal_axc_record: Any) -> None:
    changed = minimal_axc_record.copy()
    changed["ids"] = dict(minimal_axc_record["ids"])
    changed["ids"]["capsule_id"] = "axc:sha256:" + "0" * 64

    assert record_hash(minimal_axc_record) != record_hash(changed)
    assert semantic_content_hash(minimal_axc_record) == semantic_content_hash(changed)
