"""Claim-field substrate builder."""

from __future__ import annotations

import platform
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import orjson
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from hcaps import __version__
from hcaps.extraction.claims import CandidateClaim, extract_claims
from hcaps.extraction.relations import RelationCandidate, generate_relation_candidates
from hcaps.format.capsule import AxcCapsule, axc_from_claim_capsule
from hcaps.format.hashing import file_hash as axf_file_hash
from hcaps.format.package import build_package_from_axc_stream
from hcaps.format.streams import write_axc_stream
from hcaps.ingest.chunking import chunk_documents
from hcaps.ingest.documents import SourceDocument
from hcaps.ingest.readers import UNKNOWN_SOURCE_TIME, input_file_hashes, read_source_documents
from hcaps.schema.capsule import (
    Claim,
    ClaimCapsule,
    ClaimType,
    ContextFiber,
    EpistemicState,
    EvidenceType,
    LicenseStatus,
    LineageRecord,
    ProvenanceRecord,
    QualitySignals,
    Relation,
    StabilityLabel,
    SurfaceForms,
    TrainingTargets,
)
from hcaps.schema.document import DocumentSpan, TemporalCutoff
from hcaps.schema.identifiers import ClaimId, DocumentId, FamilyId
from hcaps.store.jsonl import JsonlCapsuleStore
from hcaps.substrate.canonicalize import (
    canonical_fingerprint,
    claim_similarity,
    safe_slug,
    stable_id,
)
from hcaps.substrate.manifest import BuildWarning, SubstrateBuildConfig, SubstrateBuildManifest
from hcaps.utils.hashing import file_sha256, hash_record
from hcaps.utils.time import utc_now


class ClaimFamily(BaseModel):
    """Deterministic grouping of near-equivalent claim candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family_id: FamilyId
    schema_claim_id: ClaimId
    canonical_claim_text: str = Field(min_length=1)
    canonical_fingerprint: str = Field(min_length=1)
    representative_claim_id: ClaimId
    member_claim_ids: list[ClaimId] = Field(min_length=1)
    source_document_ids: list[DocumentId] = Field(min_length=1)
    domains: list[str] = Field(default_factory=list)
    first_seen: AwareDatetime | None = None
    evidence_count: int = Field(ge=1)
    independent_source_count_proxy: int = Field(ge=1)
    confidence_summary: float = Field(ge=0.0, le=1.0)


class SubstrateBuildResult(BaseModel):
    """In-memory output summary returned by the builder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: SubstrateBuildManifest
    capsules: list[ClaimCapsule]
    axc_capsules: list[AxcCapsule]
    families: list[ClaimFamily]
    relations: list[RelationCandidate]


class ClaimFieldSubstrateBuilder:
    """Deterministic document-to-capsule substrate builder."""

    def __init__(self, config: SubstrateBuildConfig) -> None:
        self.config = config
        self.store = JsonlCapsuleStore()

    def build(self) -> SubstrateBuildResult:
        build_timestamp = utc_now()
        warnings: list[BuildWarning] = []
        documents, ingest_warnings, skipped_files = read_source_documents(
            self.config.input_path,
            self.config,
        )
        warnings.extend(ingest_warnings)

        documents, temporal_warnings, temporal_skipped = _filter_documents_by_cutoff(
            documents,
            self.config,
        )
        warnings.extend(temporal_warnings)
        skipped_files += temporal_skipped

        chunks = chunk_documents(documents, self.config)
        candidate_claims = extract_claims(chunks, documents, self.config)
        families = build_claim_families(candidate_claims, documents, self.config)
        relations = generate_relation_candidates(families, candidate_claims)
        capsules = build_claim_capsules(
            families=families,
            claims=candidate_claims,
            documents=documents,
            relations=relations,
            config=self.config,
            build_timestamp=build_timestamp,
        )

        write_result = self.store.write_capsules(self.config.output_path, capsules)
        axc_capsules = [axc_from_claim_capsule(capsule) for capsule in capsules]
        axc_output_path = _axc_output_path(self.config)
        axc_record_count = write_axc_stream(axc_output_path, axc_capsules)
        axc_output_hash = axf_file_hash(axc_output_path)
        if self.config.axp_package_path is not None:
            build_package_from_axc_stream(
                self.config.axp_package_path,
                dataset_name=self.config.dataset_name,
                capsules=axc_capsules,
                builder="claim-field-substrate-builder",
                builder_version=self.config.pipeline_version,
            )
        config_hash = hash_record(self.config.model_dump(mode="json"))
        output_hashes = {
            str(self.config.output_path): write_result.content_hash,
            str(axc_output_path): axc_output_hash,
        }
        output_paths = [
            str(self.config.output_path),
            str(axc_output_path),
            str(self.config.manifest_path),
        ]
        if self.config.axp_package_path is not None:
            output_paths.append(str(self.config.axp_package_path))
        manifest = _build_manifest(
            config=self.config,
            build_timestamp=build_timestamp,
            config_hash=config_hash,
            source_document_count=len(documents),
            chunk_count=len(chunks),
            candidate_claim_count=len(candidate_claims),
            claim_family_count=len(families),
            relation_candidate_count=len(relations),
            emitted_capsule_count=write_result.record_count,
            skipped_file_count=skipped_files,
            warnings=warnings,
            output_hashes=output_hashes,
            output_paths=output_paths,
        )
        if axc_record_count != write_result.record_count:
            msg = "AXC export count differs from internal capsule count"
            raise ValueError(msg)
        _write_build_manifest(self.config.manifest_path, manifest)
        return SubstrateBuildResult(
            manifest=manifest,
            capsules=capsules,
            axc_capsules=axc_capsules,
            families=families,
            relations=relations,
        )


def build_claim_families(
    claims: list[CandidateClaim],
    documents: list[SourceDocument],
    config: SubstrateBuildConfig,
) -> list[ClaimFamily]:
    """Group near-equivalent claim candidates deterministically."""

    if not claims:
        return []
    groups: list[list[CandidateClaim]] = []
    representatives: list[CandidateClaim] = []
    for claim in sorted(claims, key=lambda item: (item.normalized_text, item.claim_id)):
        best_index: int | None = None
        best_score = 0.0
        for index, representative in enumerate(representatives):
            score = claim_similarity(claim.text, representative.text)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= config.claim_family_similarity_threshold:
            groups[best_index].append(claim)
        else:
            groups.append([claim])
            representatives.append(claim)

    document_by_id = {document.document_id: document for document in documents}
    families = [_family_from_group(group, document_by_id) for group in groups]
    return sorted(families, key=lambda family: family.family_id)


def build_claim_capsules(
    *,
    families: list[ClaimFamily],
    claims: list[CandidateClaim],
    documents: list[SourceDocument],
    relations: list[RelationCandidate],
    config: SubstrateBuildConfig,
    build_timestamp: datetime,
) -> list[ClaimCapsule]:
    claim_by_id = {claim.claim_id: claim for claim in claims}
    document_by_id = {document.document_id: document for document in documents}
    family_by_id = {family.family_id: family for family in families}
    relations_by_source: dict[str, list[RelationCandidate]] = {}
    for relation in relations:
        relations_by_source.setdefault(relation.source_family_id, []).append(relation)

    config_hash = hash_record(config.model_dump(mode="json"))
    source_manifest_id = stable_id("manifest", config_hash)
    capsules: list[ClaimCapsule] = []
    for family in families:
        representative = claim_by_id[family.representative_claim_id]
        member_claims = [claim_by_id[claim_id] for claim_id in family.member_claim_ids]
        source_documents = [
            document_by_id[document_id] for document_id in family.source_document_ids
        ]
        cutoff_at = _cutoff_for_family(source_documents, config, build_timestamp)
        spans = [
            _span_for_claim(claim, document_by_id[claim.document_id]) for claim in member_claims
        ]
        provenance = _provenance_for_family(family, member_claims, document_by_id, cutoff_at)
        outgoing = relations_by_source.get(family.family_id, [])
        schema_relations = [
            Relation(
                relation_id=relation.relation_id,
                relation_type=relation.relation_type,
                target_claim_id=family_by_id[relation.target_family_id].schema_claim_id,
                confidence=relation.confidence,
                evidence_source_ids=[document_by_id[representative.document_id].source_id],
                rationale=relation.evidence_text,
            )
            for relation in outgoing
        ]
        input_timestamps = [record.source_timestamp for record in provenance]
        capsule_id = stable_id(
            "cap",
            family.family_id,
            family.canonical_fingerprint,
            ",".join(family.source_document_ids),
        )
        capsules.append(
            ClaimCapsule(
                capsule_id=capsule_id,
                schema_version=config.schema_version,
                created_at=build_timestamp,
                updated_at=build_timestamp,
                claim=Claim(
                    claim_id=family.schema_claim_id,
                    canonical_text=family.canonical_claim_text,
                    claim_type=_majority_claim_type(member_claims),
                    language="en",
                    scope=", ".join(family.domains) if family.domains else "unknown",
                ),
                surface_forms=SurfaceForms(
                    primary_text=family.canonical_claim_text,
                    alternate_texts=sorted(
                        {
                            claim.text
                            for claim in member_claims
                            if claim.text != family.canonical_claim_text
                        }
                    ),
                    original_spans=spans,
                ),
                epistemic_state=EpistemicState(
                    ontic_compatibility=0.5,
                    evidential_anchoring=round(
                        min(0.8, 0.3 + 0.15 * family.independent_source_count_proxy),
                        3,
                    ),
                    transformation_pressure=0.0,
                    uncertainty=round(max(0.2, 1.0 - family.confidence_summary), 3),
                    stability_label=StabilityLabel.UNKNOWN,
                    redundancy_effective_n=float(family.independent_source_count_proxy),
                    redundancy_basis=(
                        "distinct source documents carrying equivalent bootstrap claims"
                    ),
                ),
                provenance=provenance,
                relations=schema_relations,
                context=ContextFiber(
                    context_id=stable_id("ctx", family.family_id, cutoff_at.isoformat()),
                    domains=family.domains or ["unknown"],
                    temporal_cutoff=TemporalCutoff(
                        cutoff_at=cutoff_at,
                        input_source_timestamps=input_timestamps,
                        prediction_target_timestamps=[],
                        future_facing=False,
                    ),
                    community_ids=[
                        stable_id("comm", safe_slug(domain)) for domain in family.domains
                    ],
                ),
                training_targets=TrainingTargets(
                    next_token_text=family.canonical_claim_text,
                    relation_targets=[relation.relation_type for relation in outgoing],
                    provenance_targets=[record.source_id for record in provenance],
                    future_facing=False,
                ),
                quality=QualitySignals(
                    extraction_confidence=family.confidence_summary,
                    license_status=_license_status(source_documents),
                    quality_notes=(
                        "Generated by deterministic bootstrap extractor; values are "
                        "proxies for substrate construction, not benchmark claims."
                    ),
                ),
                lineage=LineageRecord(
                    pipeline_name="claim-field-substrate-builder",
                    pipeline_version=config.pipeline_version,
                    source_manifest_id=source_manifest_id,
                    parent_capsule_ids=[],
                    generation_method="deterministic-bootstrap-extractor",
                ),
            )
        )
    return capsules


def _family_from_group(
    group: list[CandidateClaim],
    document_by_id: dict[str, SourceDocument],
) -> ClaimFamily:
    representative = sorted(
        group, key=lambda claim: (-claim.confidence, len(claim.text), claim.claim_id)
    )[0]
    fingerprint = canonical_fingerprint(representative.text)
    source_document_ids = sorted({claim.document_id for claim in group})
    source_documents = [document_by_id[document_id] for document_id in source_document_ids]
    domains = sorted({domain for document in source_documents for domain in document.domains})
    first_seen_values = [
        document.published_at for document in source_documents if document.published_at
    ]
    confidence = round(sum(claim.confidence for claim in group) / len(group), 3)
    return ClaimFamily(
        family_id=stable_id("family", fingerprint),
        schema_claim_id=stable_id("claim", "family", fingerprint),
        canonical_claim_text=representative.text,
        canonical_fingerprint=fingerprint,
        representative_claim_id=representative.claim_id,
        member_claim_ids=sorted(claim.claim_id for claim in group),
        source_document_ids=source_document_ids,
        domains=domains,
        first_seen=min(first_seen_values) if first_seen_values else None,
        evidence_count=len(group),
        independent_source_count_proxy=len(source_document_ids),
        confidence_summary=confidence,
    )


def _span_for_claim(claim: CandidateClaim, document: SourceDocument) -> DocumentSpan:
    return DocumentSpan(
        span_id=stable_id("span", claim.claim_id, document.document_id),
        document_id=document.document_id,
        text=claim.text,
        start_char=claim.source_start_char,
        end_char=claim.source_end_char,
        source_timestamp=document.published_at,
    )


def _provenance_for_family(
    family: ClaimFamily,
    member_claims: list[CandidateClaim],
    document_by_id: dict[str, SourceDocument],
    cutoff_at: datetime,
) -> list[ProvenanceRecord]:
    claims_by_document: dict[str, list[CandidateClaim]] = {}
    for claim in member_claims:
        claims_by_document.setdefault(claim.document_id, []).append(claim)

    provenance: list[ProvenanceRecord] = []
    for document_id in sorted(claims_by_document):
        document = document_by_id[document_id]
        timestamp = document.published_at or min(cutoff_at, UNKNOWN_SOURCE_TIME)
        provenance.append(
            ProvenanceRecord(
                provenance_id=stable_id("src", family.family_id, document.source_id),
                source_id=document.source_id,
                document_id=document.document_id,
                source_title=document.title,
                source_timestamp=timestamp,
                evidence_type=EvidenceType.PRIMARY,
                evidence_span_ids=[
                    stable_id("span", claim.claim_id, document.document_id)
                    for claim in claims_by_document[document_id]
                ],
                source_uri=document.source_url,
                license=document.license,
            )
        )
    return provenance


def _cutoff_for_family(
    documents: list[SourceDocument],
    config: SubstrateBuildConfig,
    build_timestamp: datetime,
) -> datetime:
    config_cutoff = _config_cutoff(config)
    if config_cutoff is not None:
        return config_cutoff
    document_cutoffs = [
        document.temporal_cutoff_at for document in documents if document.temporal_cutoff_at
    ]
    if document_cutoffs:
        return min(document_cutoffs)
    return build_timestamp


def _config_cutoff(config: SubstrateBuildConfig) -> datetime | None:
    if config.cutoff_date is None:
        return None
    return datetime(
        config.cutoff_date.year, config.cutoff_date.month, config.cutoff_date.day, tzinfo=UTC
    )


def _filter_documents_by_cutoff(
    documents: list[SourceDocument],
    config: SubstrateBuildConfig,
) -> tuple[list[SourceDocument], list[BuildWarning], int]:
    cutoff = _config_cutoff(config)
    if cutoff is None:
        return documents, [], 0

    kept: list[SourceDocument] = []
    warnings: list[BuildWarning] = []
    skipped = 0
    for document in documents:
        if document.published_at and document.published_at > cutoff:
            message = (
                f"excluded post-cutoff source dated {document.published_at.date()} "
                f"after cutoff {cutoff.date()}"
            )
            warnings.append(
                BuildWarning(
                    code="post_cutoff_source_excluded", message=message, path=document.source_path
                )
            )
            skipped += 1
            if not config.strict_temporal_cutoff:
                kept.append(document)
            continue
        kept.append(document)
    return kept, warnings, skipped


def _majority_claim_type(claims: list[CandidateClaim]) -> ClaimType:
    counts = Counter(claim.claim_type for claim in claims)
    return sorted(counts, key=lambda claim_type: (-counts[claim_type], claim_type.value))[0]


def _license_status(documents: list[SourceDocument]) -> LicenseStatus:
    licenses = {document.license.casefold() for document in documents}
    if any(value in {"restricted", "proprietary"} for value in licenses):
        return LicenseStatus.RESTRICTED
    if licenses and "unknown" not in licenses:
        return LicenseStatus.KNOWN
    return LicenseStatus.UNKNOWN


def _build_manifest(
    *,
    config: SubstrateBuildConfig,
    build_timestamp: datetime,
    config_hash: str,
    source_document_count: int,
    chunk_count: int,
    candidate_claim_count: int,
    claim_family_count: int,
    relation_candidate_count: int,
    emitted_capsule_count: int,
    skipped_file_count: int,
    warnings: list[BuildWarning],
    output_hashes: dict[str, str],
    output_paths: list[str],
) -> SubstrateBuildManifest:
    joined_hashes = "|".join(f"{key}:{value}" for key, value in sorted(output_hashes.items()))
    run_id = stable_id("run", config_hash, build_timestamp.isoformat(), joined_hashes)
    return SubstrateBuildManifest(
        run_id=run_id,
        package_version=__version__,
        pipeline_version=config.pipeline_version,
        python_version=platform.python_version(),
        timestamp=build_timestamp,
        input_paths=[str(config.input_path)],
        output_paths=output_paths,
        config_hash=config_hash,
        source_document_count=source_document_count,
        chunk_count=chunk_count,
        candidate_claim_count=candidate_claim_count,
        claim_family_count=claim_family_count,
        relation_candidate_count=relation_candidate_count,
        emitted_capsule_count=emitted_capsule_count,
        skipped_file_count=skipped_file_count,
        warnings=warnings,
        input_file_hashes=input_file_hashes(config.input_path),
        output_file_hashes=output_hashes,
        temporal_cutoff=config.cutoff_date.isoformat() if config.cutoff_date else None,
    )


def _write_build_manifest(path: Path, manifest: SubstrateBuildManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = orjson.dumps(
        manifest.model_dump(mode="json"),
        option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2,
    )
    path.write_bytes(payload + b"\n")


def read_build_manifest(path: Path) -> SubstrateBuildManifest:
    return SubstrateBuildManifest.model_validate_json(path.read_bytes())


def build_substrate(config: SubstrateBuildConfig) -> SubstrateBuildResult:
    """Convenience function for one-shot substrate builds."""

    return ClaimFieldSubstrateBuilder(config).build()


def output_file_hash(path: Path) -> str:
    return file_sha256(path)


def _axc_output_path(config: SubstrateBuildConfig) -> Path:
    if config.axc_output_path is not None:
        return config.axc_output_path
    if config.output_path.suffix == ".axc":
        return config.output_path
    if config.output_path.name.endswith(".jsonl"):
        name = config.output_path.name[: -len(".jsonl")]
        return config.output_path.with_name(f"{name}.axc")
    return config.output_path.with_suffix(".axc")
