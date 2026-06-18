"""Negative candidate pool generation for later structured objectives."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from hcaps.extraction.contracts import ProviderDisagreement
from hcaps.extraction.relations import RelationCandidate
from hcaps.substrate.canonicalize import claim_similarity, stable_id, token_jaccard

if TYPE_CHECKING:
    from hcaps.substrate.builder import ClaimFamily

MIN_HARD_NEGATIVE_OVERLAP = 0.18
MAX_HARD_NEGATIVE_OVERLAP = 0.55
MIN_NEAR_DISTINCT_SIMILARITY = 0.42
MAX_NEAR_DISTINCT_SIMILARITY = 0.72
MAX_POOL_RECORDS = 200


def generate_negative_pools(
    families: list[ClaimFamily],
    relations: list[RelationCandidate],
    disagreements: Iterable[ProviderDisagreement],
) -> list[dict[str, Any]]:
    """Generate deterministic negative pools for future relation/provenance/context tasks."""

    relation_pairs = {
        (relation.source_family_id, relation.target_family_id) for relation in relations
    }
    records: list[dict[str, Any]] = []
    records.extend(_relation_hard_negatives(families, relation_pairs))
    records.extend(_same_topic_unrelated(families, relation_pairs))
    records.extend(_near_but_distinct(families))
    records.extend(_provenance_distractors(families))
    records.extend(_temporal_distractors(families))
    records.extend(_provider_disagreement_cases(disagreements))
    return sorted(records, key=lambda record: str(record["pool_id"]))


def _relation_hard_negatives(
    families: list[ClaimFamily],
    relation_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for left in families:
        for right in families:
            pair = (left.family_id, right.family_id)
            if left.family_id == right.family_id or pair in relation_pairs:
                continue
            overlap = token_jaccard(left.canonical_claim_text, right.canonical_claim_text)
            if MIN_HARD_NEGATIVE_OVERLAP <= overlap <= MAX_HARD_NEGATIVE_OVERLAP:
                records.append(
                    _record("relation_hard_negative", left.family_id, right.family_id, overlap)
                )
    return records[:MAX_POOL_RECORDS]


def _same_topic_unrelated(
    families: list[ClaimFamily],
    relation_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for left in families:
        for right in families:
            pair = (left.family_id, right.family_id)
            if left.family_id >= right.family_id or pair in relation_pairs:
                continue
            if set(left.domains) & set(right.domains):
                overlap = token_jaccard(left.canonical_claim_text, right.canonical_claim_text)
                if overlap < MIN_HARD_NEGATIVE_OVERLAP:
                    records.append(
                        _record("same_topic_unrelated", left.family_id, right.family_id, overlap)
                    )
    return records[:MAX_POOL_RECORDS]


def _near_but_distinct(families: list[ClaimFamily]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for left in families:
        for right in families:
            if left.family_id >= right.family_id:
                continue
            similarity = claim_similarity(left.canonical_claim_text, right.canonical_claim_text)
            if MIN_NEAR_DISTINCT_SIMILARITY <= similarity < MAX_NEAR_DISTINCT_SIMILARITY:
                records.append(
                    _record("near_but_distinct_claim", left.family_id, right.family_id, similarity)
                )
    return records[:MAX_POOL_RECORDS]


def _provenance_distractors(families: list[ClaimFamily]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for left in families:
        for right in families:
            if left.family_id == right.family_id:
                continue
            left_sources = set(left.source_document_ids)
            right_sources = set(right.source_document_ids)
            if not left_sources & right_sources:
                records.append(
                    {
                        "pool_id": stable_id("neg", "provenance", left.family_id, right.family_id),
                        "pool_type": "provenance_distractor_source",
                        "source_family_id": left.family_id,
                        "distractor_family_id": right.family_id,
                        "distractor_document_ids": sorted(right_sources),
                    }
                )
    return records[:MAX_POOL_RECORDS]


def _temporal_distractors(families: list[ClaimFamily]) -> list[dict[str, Any]]:
    dated = [family for family in families if family.first_seen is not None]
    records: list[dict[str, Any]] = []
    for left in dated:
        for right in dated:
            if (
                left.family_id == right.family_id
                or left.first_seen is None
                or right.first_seen is None
            ):
                continue
            if right.first_seen > left.first_seen:
                records.append(
                    {
                        "pool_id": stable_id("neg", "temporal", left.family_id, right.family_id),
                        "pool_type": "temporal_distractor",
                        "source_family_id": left.family_id,
                        "later_family_id": right.family_id,
                        "source_first_seen": left.first_seen.isoformat(),
                        "later_first_seen": right.first_seen.isoformat(),
                    }
                )
    return records[:MAX_POOL_RECORDS]


def _provider_disagreement_cases(
    disagreements: Iterable[ProviderDisagreement],
) -> list[dict[str, Any]]:
    return [
        {
            "pool_id": stable_id("neg", "provider_disagreement", item.merged_output_hash, index),
            "pool_type": "provider_disagreement_case",
            "local_output_hash": item.local_output_hash,
            "remote_output_hash": item.remote_output_hash,
            "human_review_required": item.human_review_required,
        }
        for index, item in enumerate(disagreements)
    ]


def _record(
    pool_type: str,
    source_family_id: str,
    target_family_id: str,
    score: float,
) -> dict[str, Any]:
    return {
        "pool_id": stable_id("neg", pool_type, source_family_id, target_family_id),
        "pool_type": pool_type,
        "source_family_id": source_family_id,
        "target_family_id": target_family_id,
        "score": round(score, 3),
    }
