from __future__ import annotations

from tests.spec._helpers import assert_contains_all, read_spec

NEGATIVE_TYPES = [
    "near_topic_negative",
    "near_claim_negative",
    "same_context_unrelated",
    "same_source_unrelated",
    "contradiction_candidate",
    "supersession_candidate",
    "temporal_negative",
    "provider_disagreement_negative",
    "hard_relation_negative",
    "provenance_negative",
]


def test_negative_sampling_types_and_metadata_are_registered() -> None:
    text = read_spec("NEGATIVE_SAMPLING_V1.md")

    assert_contains_all(
        text,
        [
            *NEGATIVE_TYPES,
            "negative_type",
            "hardness_score",
            "sampling_method",
            "sampling_basis",
            "sampling_seed",
            "provider_disagreement_ref",
            "temporal_validity",
            "split_eligibility",
        ],
    )


def test_vocabulary_registry_contains_negative_sample_types() -> None:
    text = read_spec("VOCABULARY_REGISTRY_V1.md")

    assert_contains_all(text, NEGATIVE_TYPES)
