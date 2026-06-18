from __future__ import annotations

from copy import deepcopy

from hcaps.falsification.controls import shuffle_contexts


def _capsules() -> list[dict[str, object]]:
    return [
        {"capsule_id": f"cap_{index}", "context": {"context_id": f"ctx_{index}"}}
        for index in range(4)
    ]


def _context_order(capsules: list[dict[str, object]]) -> list[str]:
    return [str(capsule["context"]["context_id"]) for capsule in capsules]  # type: ignore[index]


def test_context_shuffle_is_deterministic_with_same_seed() -> None:
    capsules = _capsules()

    assert shuffle_contexts(capsules, seed=13) == shuffle_contexts(capsules, seed=13)


def test_context_shuffle_changes_order_with_different_seed_when_possible() -> None:
    capsules = _capsules()

    assert _context_order(shuffle_contexts(capsules, seed=13)) != _context_order(
        shuffle_contexts(capsules, seed=14)
    )


def test_context_shuffle_preserves_record_count_and_originals() -> None:
    capsules = _capsules()
    original = deepcopy(capsules)

    shuffled = shuffle_contexts(capsules, seed=13)

    assert len(shuffled) == len(capsules)
    assert capsules == original
