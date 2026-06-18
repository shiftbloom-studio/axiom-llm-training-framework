from __future__ import annotations

from pathlib import Path

import pytest

from hcaps.store.base import InvalidCapsuleRecordError
from hcaps.store.jsonl import JsonlCapsuleStore
from hcaps.utils.hashing import hash_record

FIXTURES = Path(__file__).parent / "fixtures"


def test_invalid_fixture_fails_with_useful_error() -> None:
    store = JsonlCapsuleStore()

    with pytest.raises(InvalidCapsuleRecordError, match=r"invalid_capsules\.jsonl:1"):
        list(store.read_capsules(FIXTURES / "invalid_capsules.jsonl"))


def test_validate_collects_invalid_errors() -> None:
    report = JsonlCapsuleStore().validate(FIXTURES / "invalid_capsules.jsonl")

    assert not report.ok
    assert report.valid_count == 0
    assert "provenance" in report.errors[0]


def test_jsonl_roundtrip_preserves_capsule_ids_and_hashes(tmp_path: Path) -> None:
    store = JsonlCapsuleStore()
    capsules = list(store.read_capsules(FIXTURES / "valid_capsules.jsonl"))
    before_hashes = [hash_record(capsule.model_dump(mode="json")) for capsule in capsules]

    output = tmp_path / "roundtrip.jsonl"
    result = store.write_capsules(output, capsules)
    roundtripped = list(store.read_capsules(output))
    after_hashes = [hash_record(capsule.model_dump(mode="json")) for capsule in roundtripped]

    assert result.record_count == 2
    assert [capsule.capsule_id for capsule in roundtripped] == [
        "cap_ulcer_hpylori_001",
        "cap_aspirin_cox_001",
    ]
    assert after_hashes == before_hashes
