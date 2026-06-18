"""Canonical Axiom claim-state data contract.

The `HoloCapsule` model name is legacy internal compatibility. Public artifacts
and documentation use Axiom, Claim-State Capsule, AXC, AXP, AXT, and AXC-out.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from hcaps.schema.document import DocumentSpan, TemporalCutoff
from hcaps.schema.identifiers import (
    SCHEMA_VERSION_PATTERN,
    CapsuleId,
    ClaimId,
    CommunityId,
    ContextId,
    DocumentId,
    ManifestId,
    RelationId,
    SourceId,
    SpanId,
)
from hcaps.schema.validators import require_future_targets_after_cutoff, require_not_after_cutoff

NonEmptyString = Annotated[str, Field(min_length=1)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class ClaimType(StrEnum):
    """High-level claim classes used by the training substrate."""

    SCIENTIFIC_CLAIM = "scientific_claim"
    CAUSAL_CLAIM = "causal_claim"
    MEASUREMENT_CLAIM = "measurement_claim"
    METHOD_CLAIM = "method_claim"
    DEFINITIONAL_CLAIM = "definitional_claim"
    HISTORICAL_CLAIM = "historical_claim"
    OTHER = "other"


class StabilityLabel(StrEnum):
    """Epistemic stability states, not truth labels."""

    EMERGING = "emerging"
    CONTESTED = "contested"
    ESTABLISHED = "established"
    DECLINING = "declining"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"


class EvidenceType(StrEnum):
    """Provenance evidence classes."""

    PRIMARY = "primary"
    REVIEW = "review"
    REPLICATION = "replication"
    CONTRADICTION = "contradiction"
    BACKGROUND = "background"
    SYNTHETIC_DERIVED = "synthetic_derived"


class RelationType(StrEnum):
    """Typed relation labels. Free-form relation strings are intentionally rejected."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    REFINES = "refines"
    SUPERSEDES = "supersedes"
    IS_REPLICATED_BY = "is_replicated_by"
    USES_METHOD_FROM = "uses_method_from"
    SHARES_EVIDENCE_WITH = "shares_evidence_with"
    SAME_CLAIM_FAMILY_AS = "same_claim_family_as"
    NEAR_BUT_DISTINCT_FROM = "near_but_distinct_from"
    MENTIONS = "mentions"
    RELATED = "related"


class LicenseStatus(StrEnum):
    """Coarse license clarity status for downstream filtering."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    RESTRICTED = "restricted"
    SYNTHETIC = "synthetic"


class Claim(BaseModel):
    """Normalized claim identity separate from textual surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: ClaimId
    canonical_text: NonEmptyString
    claim_type: ClaimType
    language: str = Field(min_length=2, max_length=16)
    scope: str | None = Field(default=None, min_length=1)


class SurfaceForms(BaseModel):
    """Textual surfaces through which the same claim can appear."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_text: str | None = Field(default=None, min_length=1)
    alternate_texts: list[NonEmptyString] = Field(default_factory=list)
    original_spans: list[DocumentSpan] = Field(default_factory=list)
    summary: str | None = Field(default=None, min_length=1)
    teaching_note: str | None = Field(default=None, min_length=1)
    counterargument: str | None = Field(default=None, min_length=1)
    table_form: dict[str, str] | None = None

    @model_validator(mode="after")
    def require_surface(self) -> SurfaceForms:
        has_text = bool(
            self.primary_text
            or self.alternate_texts
            or self.original_spans
            or self.summary
            or self.teaching_note
            or self.counterargument
            or self.table_form
        )
        if not has_text:
            msg = "a capsule must contain at least one surface form"
            raise ValueError(msg)
        return self


class EpistemicState(BaseModel):
    """Bounded epistemic signals for a claim at a temporal cutoff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ontic_compatibility: UnitFloat
    evidential_anchoring: UnitFloat
    transformation_pressure: UnitFloat
    uncertainty: UnitFloat
    stability_label: StabilityLabel
    redundancy_effective_n: float = Field(ge=0.0)
    redundancy_measure: Literal["independent_community_effective_count"] = (
        "independent_community_effective_count"
    )
    redundancy_basis: NonEmptyString

    @model_validator(mode="after")
    def reject_raw_popularity_basis(self) -> EpistemicState:
        forbidden = ("raw popularity", "citation count", "mention count", "view count")
        basis = self.redundancy_basis.casefold()
        if any(term in basis for term in forbidden):
            msg = (
                "redundancy must represent independent-community effective count, "
                "not raw popularity or raw count proxies"
            )
            raise ValueError(msg)
        return self


class ProvenanceRecord(BaseModel):
    """Evidence or source record used as model input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provenance_id: SourceId
    source_id: SourceId
    document_id: DocumentId
    source_title: NonEmptyString
    source_timestamp: AwareDatetime
    evidence_type: EvidenceType
    evidence_span_ids: list[SpanId] = Field(default_factory=list)
    source_uri: str | None = None
    license: NonEmptyString = "unknown"


class Relation(BaseModel):
    """Typed relation from this capsule's claim to another claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_id: RelationId
    relation_type: RelationType
    target_claim_id: ClaimId
    confidence: UnitFloat
    evidence_source_ids: list[SourceId] = Field(default_factory=list)
    rationale: str | None = Field(default=None, min_length=1)


class ExperimentalGeometryObservables(BaseModel):
    """Optional HKR-inspired gauge-invariant summaries.

    Raw gauge matrices are deliberately not part of this contract. Extra fields are
    forbidden so matrix-like ad hoc payloads cannot slip into reported records.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    loop_norm: float | None = Field(default=None, ge=0.0)
    trace_summary: float | None = None
    spectrum_summary: list[float] | None = None
    curvature_score: float | None = None
    context_lability: UnitFloat | None = None
    experimental: Literal[True] = True


class ContextFiber(BaseModel):
    """Contextual coordinates for the claim field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_id: ContextId
    domains: list[NonEmptyString] = Field(min_length=1)
    temporal_cutoff: TemporalCutoff
    community_ids: list[CommunityId] = Field(default_factory=list)
    embedding_keys: list[NonEmptyString] = Field(default_factory=list)
    experimental_geometry: ExperimentalGeometryObservables | None = None


class TrainingTargets(BaseModel):
    """Targets used by future training interfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    next_token_text: str | None = Field(default=None, min_length=1)
    future_summary: str | None = Field(default=None, min_length=1)
    relation_targets: list[RelationType] = Field(default_factory=list)
    provenance_targets: list[SourceId] = Field(default_factory=list)
    stability_target: StabilityLabel | None = None
    target_timestamp: AwareDatetime | None = None
    future_facing: bool = False

    @model_validator(mode="after")
    def require_target_timestamp_for_future_targets(self) -> TrainingTargets:
        has_future_target = self.future_summary is not None or self.future_facing
        if has_future_target and self.target_timestamp is None:
            msg = "future-facing training targets require target_timestamp"
            raise ValueError(msg)
        if self.future_summary is not None and not self.future_facing:
            msg = "future_summary targets must be marked future_facing"
            raise ValueError(msg)
        return self


class QualitySignals(BaseModel):
    """Quality controls that remain separate from epistemic truth status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extraction_confidence: UnitFloat
    annotation_confidence: UnitFloat | None = None
    license_status: LicenseStatus
    deduplication_group: str | None = Field(default=None, min_length=1)
    quality_notes: str | None = Field(default=None, min_length=1)


class LineageRecord(BaseModel):
    """Reproducible processing lineage for a capsule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline_name: NonEmptyString
    pipeline_version: NonEmptyString
    source_manifest_id: ManifestId
    parent_capsule_ids: list[CapsuleId] = Field(default_factory=list)
    generation_method: NonEmptyString


class HoloCapsule(BaseModel):
    """Versioned, validated Axiom claim-state capsule.

    The class name remains for legacy internal compatibility.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    capsule_id: CapsuleId
    schema_version: str = Field(pattern=SCHEMA_VERSION_PATTERN)
    created_at: AwareDatetime
    updated_at: AwareDatetime
    claim: Claim
    surface_forms: SurfaceForms
    epistemic_state: EpistemicState
    provenance: list[ProvenanceRecord] = Field(min_length=1)
    relations: list[Relation] = Field(default_factory=list)
    context: ContextFiber
    training_targets: TrainingTargets
    quality: QualitySignals
    lineage: LineageRecord

    @model_validator(mode="after")
    def validate_capsule_contract(self) -> HoloCapsule:
        if self.updated_at < self.created_at:
            msg = "updated_at must be greater than or equal to created_at"
            raise ValueError(msg)

        cutoff = self.context.temporal_cutoff.cutoff_at
        input_timestamps = [record.source_timestamp for record in self.provenance]
        input_timestamps.extend(
            span.source_timestamp
            for span in self.surface_forms.original_spans
            if span.source_timestamp is not None
        )
        require_not_after_cutoff(input_timestamps, cutoff=cutoff, label="input source")

        if self.training_targets.future_facing and self.training_targets.target_timestamp:
            require_future_targets_after_cutoff(
                [self.training_targets.target_timestamp],
                cutoff=cutoff,
                label="future-facing prediction target",
            )

        return self


ClaimCapsule = HoloCapsule
