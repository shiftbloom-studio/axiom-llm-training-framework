"""Provider-backed extraction contracts for Axiom corpus ingress."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from hcaps.schema.capsule import ClaimType, RelationType, UnitFloat

ExtractionTask = Literal[
    "claim_extraction",
    "relation_extraction",
    "epistemic_extraction",
    "structured_view_generation",
]
ProviderMode = Literal["deterministic", "local", "remote", "human", "custom"]


class SourceSpanRef(BaseModel):
    """Reference to a source span used by an extraction task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(min_length=1)


class ProviderTrace(BaseModel):
    """Trace tying normalized provider output to provider/cache/provenance metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    provider_type: str = Field(min_length=1)
    provider_family: str = Field(min_length=1)
    provider_mode: ProviderMode
    provider_model: str = Field(min_length=1)
    extraction_task: ExtractionTask
    prompt_template_version: str = Field(min_length=1)
    cache_key: str | None = None
    input_hash: str = Field(min_length=1)
    output_hash: str = Field(min_length=1)
    confidence: UnitFloat | None = None
    escalation_reason: str | None = None
    merge_decision: str | None = None


class ExtractedClaim(BaseModel):
    """Provider-normalized claim candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    claim_type: ClaimType
    confidence: UnitFloat
    source_spans: list[SourceSpanRef] = Field(default_factory=list)
    notes: str | None = None


class ClaimExtractionOutput(BaseModel):
    """Output contract for claim extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: list[ExtractedClaim] = Field(default_factory=list)
    provider_trace: ProviderTrace
    warnings: list[str] = Field(default_factory=list)


class ExtractedRelation(BaseModel):
    """Provider-normalized relation candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_claim: str = Field(min_length=1)
    target_claim: str = Field(min_length=1)
    relation_type: RelationType
    confidence: UnitFloat
    evidence_spans: list[SourceSpanRef] = Field(default_factory=list)
    candidate_hard_negatives: list[str] = Field(default_factory=list)


class RelationExtractionOutput(BaseModel):
    """Output contract for relation extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relations: list[ExtractedRelation] = Field(default_factory=list)
    provider_trace: ProviderTrace
    warnings: list[str] = Field(default_factory=list)


class EpistemicProxy(BaseModel):
    """A conservative epistemic proxy with method and confidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float | None = Field(default=None, ge=0.0)
    method: str = Field(min_length=1)
    confidence: UnitFloat
    notes: str | None = None


class EpistemicExtractionOutput(BaseModel):
    """Output contract for epistemic proxy extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ontology_compatibility: EpistemicProxy
    evidential_anchoring: EpistemicProxy
    transformation_pressure: EpistemicProxy
    uncertainty: EpistemicProxy
    independent_redundancy: EpistemicProxy
    source_count: int = Field(default=0, ge=0)
    provider_disagreement_count: int = Field(default=0, ge=0)
    relation_degree: int = Field(default=0, ge=0)
    method_diversity_proxy: float | None = Field(default=None, ge=0.0)
    source_type_diversity_proxy: float | None = Field(default=None, ge=0.0)
    benchmark_family_diversity_proxy: float | None = Field(default=None, ge=0.0)
    provider_trace: ProviderTrace
    warnings: list[str] = Field(default_factory=list)


class StructuredViews(BaseModel):
    """Structured text projections generated from source-grounded claim material."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    neutral_summary: str | None = None
    technical_summary: str | None = None
    teaching_note: str | None = None
    faq: str | None = None
    counterargument: str | None = None
    limitations: str | None = None
    historical_update_or_temporal_note: str | None = None


class StructuredViewOutput(BaseModel):
    """Output contract for structured view generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    views: StructuredViews
    source_spans_used: list[SourceSpanRef] = Field(default_factory=list)
    confidence: UnitFloat
    human_reviewed: bool = False
    provider_trace: ProviderTrace
    warnings: list[str] = Field(default_factory=list)


class ProviderDisagreement(BaseModel):
    """A preserved disagreement between provider outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    local_output_hash: str = Field(min_length=1)
    remote_output_hash: str = Field(min_length=1)
    disagreement_type: str = Field(min_length=1)
    merge_policy: str = Field(min_length=1)
    merged_output_hash: str = Field(min_length=1)
    selected_fields: list[str] = Field(default_factory=list)
    rejected_fields: list[str] = Field(default_factory=list)
    human_review_required: bool = False


class ExtractionTraceRecord(BaseModel):
    """Serializable trace record written into P1 corpus artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1)
    created_at: AwareDatetime
    provider_trace: ProviderTrace
    task_input_hash: str = Field(min_length=1)
    normalized_output_hash: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
