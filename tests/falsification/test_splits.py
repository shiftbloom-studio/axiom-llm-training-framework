from __future__ import annotations

from hcaps.falsification.splits import (
    apply_split,
    make_deterministic_random_split,
    make_temporal_holdout,
)


def _capsules() -> list[dict[str, object]]:
    return [
        {
            "capsule_id": "cap_a",
            "context": {"temporal_cutoff": {"cutoff_at": "2020-01-01T00:00:00Z"}},
        },
        {
            "capsule_id": "cap_b",
            "context": {"temporal_cutoff": {"cutoff_at": "2022-01-01T00:00:00Z"}},
        },
        {
            "capsule_id": "cap_c",
            "context": {"temporal_cutoff": {"cutoff_at": "2024-01-01T00:00:00Z"}},
        },
    ]


def test_deterministic_random_split_is_reproducible() -> None:
    capsules = _capsules()

    split_a = make_deterministic_random_split(capsules, {"train": 0.67, "test": 0.33}, seed=7)
    split_b = make_deterministic_random_split(capsules, {"train": 0.67, "test": 0.33}, seed=7)

    assert split_a == split_b


def test_temporal_holdout_respects_cutoff_date() -> None:
    split = make_temporal_holdout(_capsules(), "2021-01-01T00:00:00Z")

    assert [capsule["capsule_id"] for capsule in split["train"]] == ["cap_a"]
    assert [capsule["capsule_id"] for capsule in split["holdout"]] == ["cap_b", "cap_c"]


def test_apply_split_selects_expected_ids() -> None:
    selected = apply_split(_capsules(), {"cap_a", "cap_c"})

    assert [capsule["capsule_id"] for capsule in selected] == ["cap_a", "cap_c"]
