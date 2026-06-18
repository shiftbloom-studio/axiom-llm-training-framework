"""Claim and relation extraction interfaces."""

from hcaps.extraction.claims import (
    CandidateClaim,
    ClaimExtractor,
    DeterministicClaimExtractor,
    extract_claims,
)
from hcaps.extraction.relations import RelationCandidate, generate_relation_candidates

__all__ = [
    "CandidateClaim",
    "ClaimExtractor",
    "DeterministicClaimExtractor",
    "RelationCandidate",
    "extract_claims",
    "generate_relation_candidates",
]
