"""Deterministic relation candidate generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from hcaps.extraction.claims import CandidateClaim
from hcaps.schema.capsule import RelationType, UnitFloat
from hcaps.schema.identifiers import ClaimId, FamilyId, RelationId
from hcaps.substrate.canonicalize import claim_similarity, stable_id, token_jaccard, token_set

if TYPE_CHECKING:
    from hcaps.substrate.builder import ClaimFamily


class RelationCandidate(BaseModel):
    """A conservative deterministic relation candidate between claim families."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_id: RelationId
    source_family_id: FamilyId
    source_claim_id: ClaimId
    target_family_id: FamilyId
    target_claim_id: ClaimId
    relation_type: RelationType
    confidence: UnitFloat
    evidence_claim_id: ClaimId
    evidence_text: str = Field(min_length=1)
    extraction_method: str = "deterministic-relation-heuristics-v0.2"


SUPPORT_CUES = ("supports", "confirms", "replicates", "is consistent with")
CONTRADICTION_CUES = ("contradicts", "fails to replicate", "challenges", "inconsistent with")
SUPERSESSION_CUES = ("replaces", "supersedes", "later work", "updates")
EXTENSION_CUES = ("extends", "generalizes", "adds", "builds on")
MENTION_CUES = ("mentions", "cites", "describes")
MIN_EVIDENCE_OVERLAP = 0.2
MIN_LEXICAL_OVERLAP = 0.18
RELATED_LEXICAL_OVERLAP = 0.28
RELATED_SIMILARITY = 0.44
RELATED_BASE_CONFIDENCE = 0.28
MAX_RELATED_CONFIDENCE = 0.46

RELATION_RULES: tuple[tuple[tuple[str, ...], RelationType, float], ...] = (
    (CONTRADICTION_CUES, RelationType.CONTRADICTS, 0.52),
    (SUPPORT_CUES, RelationType.SUPPORTS, 0.55),
    (SUPERSESSION_CUES, RelationType.SUPERSEDES, 0.5),
    (EXTENSION_CUES, RelationType.EXTENDS, 0.48),
    (MENTION_CUES, RelationType.MENTIONS, 0.42),
)


def generate_relation_candidates(
    families: Sequence[ClaimFamily],
    claims: list[CandidateClaim],
) -> list[RelationCandidate]:
    """Generate deterministic low-confidence relation candidates."""

    claim_by_id = {claim.claim_id: claim for claim in claims}
    relations: list[RelationCandidate] = []
    ordered = sorted(families, key=lambda family: family.family_id)
    for source in ordered:
        source_claim = claim_by_id[source.representative_claim_id]
        for target in ordered:
            if source.family_id == target.family_id:
                continue
            relation_type, confidence = _relation_type_for(
                source_claim.text,
                source.canonical_claim_text,
                target.canonical_claim_text,
            )
            if relation_type is None:
                continue
            relation_id = stable_id(
                "rel",
                source.family_id,
                target.family_id,
                relation_type.value,
                source_claim.claim_id,
            )
            relations.append(
                RelationCandidate(
                    relation_id=relation_id,
                    source_family_id=source.family_id,
                    source_claim_id=source.schema_claim_id,
                    target_family_id=target.family_id,
                    target_claim_id=target.schema_claim_id,
                    relation_type=relation_type,
                    confidence=confidence,
                    evidence_claim_id=source_claim.claim_id,
                    evidence_text=source_claim.text,
                )
            )
    return _deduplicate_relations(relations)


def _relation_type_for(
    evidence_text: str,
    source_text: str,
    target_text: str,
) -> tuple[RelationType | None, float]:
    lowered = evidence_text.casefold()
    lexical_overlap = token_jaccard(source_text, target_text)
    target_tokens = token_set(target_text)
    evidence_overlap = len(token_set(evidence_text) & target_tokens) / max(len(target_tokens), 1)

    if evidence_overlap < MIN_EVIDENCE_OVERLAP and lexical_overlap < MIN_LEXICAL_OVERLAP:
        return None, 0.0

    for cues, relation_type, confidence in RELATION_RULES:
        if any(cue in lowered for cue in cues):
            return relation_type, confidence

    if (
        lexical_overlap >= RELATED_LEXICAL_OVERLAP
        or claim_similarity(source_text, target_text) >= RELATED_SIMILARITY
    ):
        return (
            RelationType.RELATED,
            round(min(MAX_RELATED_CONFIDENCE, RELATED_BASE_CONFIDENCE + lexical_overlap), 3),
        )
    return None, 0.0


def _deduplicate_relations(relations: list[RelationCandidate]) -> list[RelationCandidate]:
    best_by_key: dict[tuple[str, str, RelationType], RelationCandidate] = {}
    for relation in relations:
        key = (relation.source_family_id, relation.target_family_id, relation.relation_type)
        current = best_by_key.get(key)
        if current is None or relation.confidence > current.confidence:
            best_by_key[key] = relation
    return sorted(best_by_key.values(), key=lambda relation: relation.relation_id)
