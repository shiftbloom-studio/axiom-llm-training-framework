from __future__ import annotations

from copy import deepcopy

from hcaps.falsification.controls import remove_context, remove_provenance, remove_relations


def _capsule() -> dict[str, object]:
    return {
        "capsule_id": "cap_test",
        "provenance": [{"source_id": "src_test"}],
        "relations": [{"relation_type": "supports"}],
        "context": {"context_id": "ctx_test"},
    }


def test_remove_provenance_removes_provenance_without_mutation() -> None:
    capsule = _capsule()
    original = deepcopy(capsule)

    transformed = remove_provenance(capsule)

    assert "provenance" not in transformed
    assert capsule == original


def test_remove_relations_removes_relations_without_mutation() -> None:
    capsule = _capsule()
    original = deepcopy(capsule)

    transformed = remove_relations(capsule)

    assert transformed["relations"] == []
    assert capsule == original


def test_remove_context_removes_context_without_mutation() -> None:
    capsule = _capsule()
    original = deepcopy(capsule)

    transformed = remove_context(capsule)

    assert "context" not in transformed
    assert capsule == original
