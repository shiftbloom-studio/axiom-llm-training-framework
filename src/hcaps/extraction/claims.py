"""Deterministic bootstrap claim extraction."""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from hcaps.ingest.documents import DocumentChunk, SourceDocument
from hcaps.schema.capsule import ClaimType, UnitFloat
from hcaps.schema.identifiers import ChunkId, ClaimId, DocumentId
from hcaps.substrate.canonicalize import canonical_fingerprint, stable_id
from hcaps.substrate.manifest import SubstrateBuildConfig

SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
BOILERPLATE_CUES = (
    "copyright",
    "all rights reserved",
    "references",
    "bibliography",
    "acknowledgement",
    "acknowledgment",
)
ASSERTIVE_CUES = (
    " is ",
    " are ",
    " was ",
    " were ",
    " has ",
    " have ",
    " shows ",
    " showed ",
    " suggests ",
    " indicates ",
    " demonstrates ",
    " exhibits ",
    " exhibit ",
    " asserts ",
    " supports ",
    " contradicts ",
    " causes ",
    " contributes ",
    " improves ",
    " inhibits ",
    " reduces ",
    " increases ",
    " replaces ",
    " supersedes ",
    " uses ",
)


class CandidateClaim(BaseModel):
    """A deterministic claim candidate extracted from a source chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: ClaimId
    document_id: DocumentId
    chunk_id: ChunkId
    text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    claim_type: ClaimType
    cue_category: str = Field(min_length=1)
    confidence: UnitFloat
    source_start_char: int = Field(ge=0)
    source_end_char: int = Field(gt=0)
    section_path: list[str] = Field(default_factory=list)
    extraction_method: str = "deterministic-bootstrap-v0.2"


class ClaimExtractor(Protocol):
    """Protocol for future neural, human-curated, or LLM-assisted extractors."""

    def extract(
        self,
        chunk: DocumentChunk,
        document: SourceDocument,
        config: SubstrateBuildConfig,
    ) -> list[CandidateClaim]: ...


class DeterministicClaimExtractor:
    """Transparent rule-based bootstrap extractor.

    This extractor is intentionally conservative and should be treated as a
    replaceable bootstrap component, not as high-quality claim understanding.
    """

    def extract(
        self,
        chunk: DocumentChunk,
        document: SourceDocument,
        config: SubstrateBuildConfig,
    ) -> list[CandidateClaim]:
        claims: list[CandidateClaim] = []
        for match in SENTENCE_PATTERN.finditer(chunk.text):
            sentence = " ".join(match.group(0).strip().split())
            if not _is_candidate_sentence(sentence, config):
                continue
            claim_type, cue_category, cue_bonus = _classify_sentence(sentence)
            absolute_start = chunk.start_char + match.start()
            absolute_end = absolute_start + len(match.group(0).strip())
            fingerprint = canonical_fingerprint(sentence)
            claim_id = stable_id("claim", document.document_id, chunk.chunk_id, fingerprint)
            confidence = min(0.86, 0.45 + cue_bonus + min(len(fingerprint) / 500.0, 0.12))
            claims.append(
                CandidateClaim(
                    claim_id=claim_id,
                    document_id=document.document_id,
                    chunk_id=chunk.chunk_id,
                    text=sentence,
                    normalized_text=fingerprint,
                    claim_type=claim_type,
                    cue_category=cue_category,
                    confidence=round(confidence, 3),
                    source_start_char=absolute_start,
                    source_end_char=absolute_end,
                    section_path=chunk.section_path,
                )
            )
        return claims


def extract_claims(
    chunks: list[DocumentChunk],
    documents: list[SourceDocument],
    config: SubstrateBuildConfig,
    extractor: ClaimExtractor | None = None,
) -> list[CandidateClaim]:
    active_extractor: ClaimExtractor = extractor or DeterministicClaimExtractor()
    document_by_id = {document.document_id: document for document in documents}
    claims: list[CandidateClaim] = []
    for chunk in chunks:
        claims.extend(active_extractor.extract(chunk, document_by_id[chunk.document_id], config))
    return claims


def _is_candidate_sentence(sentence: str, config: SubstrateBuildConfig) -> bool:
    lowered = sentence.casefold()
    if len(sentence) < config.min_claim_chars or len(sentence) > config.max_claim_chars:
        return False
    if sentence.endswith("?"):
        return False
    if any(cue in lowered for cue in BOILERPLATE_CUES):
        return False
    return any(cue in f" {lowered} " for cue in ASSERTIVE_CUES)


CLASSIFICATION_RULES: tuple[tuple[tuple[str, ...], ClaimType, str, float], ...] = (
    (
        (" causes ", " contributes ", " leads to ", " results in "),
        ClaimType.CAUSAL_CLAIM,
        "causal",
        0.16,
    ),
    (
        (" method ", " protocol ", " algorithm ", " uses ", " measures "),
        ClaimType.METHOD_CLAIM,
        "method",
        0.12,
    ),
    (
        (" observed ", " measured ", " increased ", " reduced ", " associated "),
        ClaimType.MEASUREMENT_CLAIM,
        "empirical",
        0.12,
    ),
    (
        (" is a ", " are a ", " refers to ", " defined as "),
        ClaimType.DEFINITIONAL_CLAIM,
        "definition",
        0.1,
    ),
    (
        (" later ", " previous ", " before ", " after ", " supersedes "),
        ClaimType.HISTORICAL_CLAIM,
        "temporal",
        0.1,
    ),
    (
        (" more than ", " less than ", " compared with ", " improves "),
        ClaimType.SCIENTIFIC_CLAIM,
        "comparison",
        0.1,
    ),
    (
        (" limitation ", " however ", " inconsistent with "),
        ClaimType.SCIENTIFIC_CLAIM,
        "limitation",
        0.08,
    ),
)


def _classify_sentence(sentence: str) -> tuple[ClaimType, str, float]:
    lowered = f" {sentence.casefold()} "
    for cues, claim_type, category, bonus in CLASSIFICATION_RULES:
        if any(cue in lowered for cue in cues):
            return claim_type, category, bonus
    return ClaimType.SCIENTIFIC_CLAIM, "assertive", 0.06
