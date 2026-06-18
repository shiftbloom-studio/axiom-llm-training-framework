"""AXC v0.1 claim-state capsule schema and adapters."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from hcaps.format.identifiers import (
    AxcCapsuleId,
    ClaimFamilyId,
    ClaimStateId,
    ContextId,
    RelationId,
    SourceId,
    SpanId,
    axc_id,
    claim_family_id,
    claim_state_id,
    context_id,
    relation_id,
    slugify,
    source_id,
    span_id,
)
from hcaps.schema.capsule import ClaimCapsule

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class ClaimStatus(StrEnum):
    UNASSESSED = "unassessed"
    EMERGING = "emerging"
    CONTESTED = "contested"
    SUPPORTED = "supported"
    ESTABLISHED = "established"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    FRAGMENTED = "fragmented"
    FIELD_DEPENDENT = "field_dependent"
    HISTORICALLY_PLAUSIBLE = "historically_plausible"


class LeakageValidationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNCHECKED = "unchecked"


class RedundancyMethod(StrEnum):
    INDEPENDENT_SOURCE_PROXY = "independent_source_proxy"
    INDEPENDENT_COMMUNITY_EFFECTIVE_COUNT = "independent_community_effective_count"
    UNKNOWN = "unknown"


class AxcIds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capsule_id: AxcCapsuleId
    claim_family_id: ClaimFamilyId
    claim_state_id: ClaimStateId
    context_id: ContextId


class ClaimIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_text: str = Field(min_length=1)
    claim_type: str = Field(min_length=1)
    language: str = Field(default="en", min_length=2, max_length=16)
    family_label: str | None = Field(default=None, min_length=1)


class SourceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    span_id: SpanId
    source_id: SourceId
    text: str = Field(min_length=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, gt=0)
    source_date: AwareDatetime | None = None
    target_only: bool = False

    @model_validator(mode="after")
    def validate_offsets(self) -> SourceSpan:
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char <= self.start_char
        ):
            msg = "end_char must be greater than start_char"
            raise ValueError(msg)
        return self


class SurfaceForms(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical: str = Field(min_length=1)
    source_spans: list[SourceSpan] = Field(default_factory=list)
    normalized_views: dict[str, str] = Field(default_factory=dict)
    generated_views: dict[str, str] = Field(default_factory=dict)


class TemporalScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid_as_of: AwareDatetime
    observed_at: AwareDatetime | None = None
    constructed_at: AwareDatetime
    source_publication_date: AwareDatetime | None = None
    cutoff_policy: str = Field(default="predictor_side_sources_must_not_exceed_valid_as_of")
    leakage_validation_status: LeakageValidationStatus = LeakageValidationStatus.UNCHECKED


class ContextScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    domains: list[str] = Field(default_factory=list)
    communities: list[str] = Field(default_factory=list)
    description: str | None = None


class ProbabilityMeasure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: UnitFloat
    confidence: UnitFloat | None = None
    method: str = Field(default="unspecified", min_length=1)


class RedundancyMeasure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    effective_count: NonNegativeFloat
    method: RedundancyMethod
    confidence: UnitFloat | None = None
    notes: str | None = None


class EpistemicState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ClaimStatus = ClaimStatus.UNASSESSED
    ontology_compatibility: ProbabilityMeasure | None = None
    evidential_anchoring: ProbabilityMeasure | None = None
    transformation_pressure: ProbabilityMeasure | None = None
    uncertainty: ProbabilityMeasure | None = None
    independent_redundancy: RedundancyMeasure
    status_trajectory: list[ClaimStatus] = Field(default_factory=list)


class AxcRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_id: RelationId
    target_claim_family_id: ClaimFamilyId
    relation_type: str = Field(min_length=1)
    confidence: UnitFloat | None = None
    evidence_span_ids: list[SpanId] = Field(default_factory=list)
    extraction_method: str | None = None


class GeometryObservables(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    gauge_policy: Literal["gauge_invariant_observables_only"] = "gauge_invariant_observables_only"
    curvature_score: float | None = None
    transport_observables: dict[str, float] = Field(default_factory=dict)
    loop_identifiers: list[str] = Field(default_factory=list)
    context_sequence: list[ContextId] = Field(default_factory=list)


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: SourceId
    title: str = Field(min_length=1)
    path: str | None = None
    url: str | None = None
    source_date: AwareDatetime | None = None
    license: str = "unknown"
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    target_only: bool = False


class ProvenanceBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: list[SourceReference] = Field(min_length=1)
    construction_method: str = Field(min_length=1)
    extractor: str | None = None


class TrainingBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    eligible: bool = True
    allowed_splits: list[str] = Field(default_factory=lambda: ["train", "validation", "test"])
    target_fields: list[str] = Field(default_factory=list)
    future_label_fields: list[str] = Field(default_factory=list)


class QualityBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    extraction_confidence: UnitFloat | None = None
    validation_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AxcCapsule(BaseModel):
    """AXC v0.1 claim-state capsule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["AXC"] = "AXC"
    format_version: Literal["0.1.0"] = "0.1.0"
    ids: AxcIds
    claim: ClaimIdentity
    surface_forms: SurfaceForms
    temporal: TemporalScope
    context: ContextScope
    epistemic_state: EpistemicState
    relations: list[AxcRelation] = Field(default_factory=list)
    geometry: GeometryObservables = Field(default_factory=GeometryObservables)
    provenance: ProvenanceBlock
    training: TrainingBlock = Field(default_factory=TrainingBlock)
    quality: QualityBlock = Field(default_factory=QualityBlock)

    @model_validator(mode="after")
    def validate_temporal_scope(self) -> AxcCapsule:
        valid_as_of = self.temporal.valid_as_of
        source_dates: list[tuple[str, AwareDatetime, bool]] = []
        if self.temporal.source_publication_date is not None:
            source_dates.append(
                ("temporal.source_publication_date", self.temporal.source_publication_date, False)
            )
        for source in self.provenance.sources:
            if source.source_date is not None:
                source_dates.append((source.source_id, source.source_date, source.target_only))
        for span in self.surface_forms.source_spans:
            if span.source_date is not None:
                source_dates.append((span.span_id, span.source_date, span.target_only))
        for label, source_date, target_only in source_dates:
            if source_date > valid_as_of and not target_only:
                msg = f"temporal leakage: {label} source_date is after valid_as_of"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_provenance_references(self) -> AxcCapsule:
        source_ids = {source.source_id for source in self.provenance.sources}
        span_ids = {span.span_id for span in self.surface_forms.source_spans}
        for span in self.surface_forms.source_spans:
            if span.source_id not in source_ids:
                msg = f"provenance reference missing for source_id {span.source_id}"
                raise ValueError(msg)
        for relation in self.relations:
            for span_id_value in relation.evidence_span_ids:
                if span_id_value not in span_ids:
                    msg = (
                        "relation evidence_span_ids must reference surface_forms.source_spans: "
                        f"{span_id_value}"
                    )
                    raise ValueError(msg)
        return self


def axc_from_claim_capsule(capsule: ClaimCapsule) -> AxcCapsule:
    """Adapt the Step 01/02 internal capsule model into canonical AXC."""

    valid_as_of = capsule.context.temporal_cutoff.cutoff_at
    claim_slug = slugify(capsule.claim.canonical_text)[:48]
    family_id = claim_family_id(claim_slug, capsule.claim.claim_id, capsule.claim.canonical_text)
    state_id = claim_state_id(valid_as_of.date().isoformat(), capsule.capsule_id, family_id)
    axc_capsule_id = axc_id(family_id, state_id, capsule.claim.canonical_text)
    provenance_sources = [
        SourceReference(
            source_id=source_id("document", record.source_id, record.document_id),
            title=record.source_title,
            url=record.source_uri,
            source_date=record.source_timestamp,
            license=record.license,
        )
        for record in capsule.provenance
    ]
    source_id_by_document = {
        record.document_id: source_id("document", record.source_id, record.document_id)
        for record in capsule.provenance
    }
    source_spans = [
        SourceSpan(
            span_id=span_id(span.span_id, span.document_id),
            source_id=source_id_by_document.get(
                span.document_id,
                source_id("document", span.document_id),
            ),
            text=span.text,
            start_char=span.start_char,
            end_char=span.end_char,
            source_date=span.source_timestamp,
        )
        for span in capsule.surface_forms.original_spans
    ]
    axc_relations = [
        AxcRelation(
            relation_id=relation_id(relation.relation_id, relation.target_claim_id),
            target_claim_family_id=claim_family_id(
                relation.target_claim_id,
                relation.target_claim_id,
            ),
            relation_type=relation.relation_type.value,
            confidence=relation.confidence,
            evidence_span_ids=[],
            extraction_method="step02-adapter",
        )
        for relation in capsule.relations
    ]
    return AxcCapsule(
        ids=AxcIds(
            capsule_id=axc_capsule_id,
            claim_family_id=family_id,
            claim_state_id=state_id,
            context_id=context_id(",".join(capsule.context.domains) or "unknown"),
        ),
        claim=ClaimIdentity(
            canonical_text=capsule.claim.canonical_text,
            claim_type=capsule.claim.claim_type.value,
            language=capsule.claim.language,
            family_label=capsule.claim.scope,
        ),
        surface_forms=SurfaceForms(
            canonical=capsule.claim.canonical_text,
            source_spans=source_spans,
            normalized_views={
                "primary": capsule.surface_forms.primary_text or capsule.claim.canonical_text
            },
            generated_views={
                key: value
                for key, value in {
                    "summary": capsule.surface_forms.summary,
                    "teaching_note": capsule.surface_forms.teaching_note,
                    "counterargument": capsule.surface_forms.counterargument,
                }.items()
                if value is not None
            },
        ),
        temporal=TemporalScope(
            valid_as_of=valid_as_of,
            observed_at=valid_as_of,
            constructed_at=capsule.created_at,
            source_publication_date=min(record.source_timestamp for record in capsule.provenance),
            leakage_validation_status=LeakageValidationStatus.PASSED,
        ),
        context=ContextScope(
            domains=capsule.context.domains,
            communities=capsule.context.community_ids,
        ),
        epistemic_state=EpistemicState(
            status=ClaimStatus.UNASSESSED,
            ontology_compatibility=ProbabilityMeasure(
                value=capsule.epistemic_state.ontic_compatibility,
                method="step01-proxy",
            ),
            evidential_anchoring=ProbabilityMeasure(
                value=capsule.epistemic_state.evidential_anchoring,
                method="step01-proxy",
            ),
            transformation_pressure=ProbabilityMeasure(
                value=capsule.epistemic_state.transformation_pressure,
                method="step01-proxy",
            ),
            uncertainty=ProbabilityMeasure(
                value=capsule.epistemic_state.uncertainty,
                method="step01-proxy",
            ),
            independent_redundancy=RedundancyMeasure(
                effective_count=capsule.epistemic_state.redundancy_effective_n,
                method=RedundancyMethod.INDEPENDENT_COMMUNITY_EFFECTIVE_COUNT,
                notes=capsule.epistemic_state.redundancy_basis,
            ),
        ),
        relations=axc_relations,
        provenance=ProvenanceBlock(
            sources=provenance_sources,
            construction_method=capsule.lineage.generation_method,
            extractor=capsule.lineage.pipeline_name,
        ),
        training=TrainingBlock(
            eligible=True,
            target_fields=["next_token_text"] if capsule.training_targets.next_token_text else [],
            future_label_fields=(
                ["future_summary"] if capsule.training_targets.future_summary else []
            ),
        ),
        quality=QualityBlock(
            extraction_confidence=capsule.quality.extraction_confidence,
            validation_notes=[capsule.quality.quality_notes]
            if capsule.quality.quality_notes
            else [],
        ),
    )
