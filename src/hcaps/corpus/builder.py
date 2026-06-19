"""Provider-aware corpus builder for Axiom P1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from hcaps.corpus.gold import gold_candidate_records
from hcaps.corpus.manifest import CorpusBuildConfig, CorpusBuildResult, CorpusManifest
from hcaps.corpus.quality import quality_tier
from hcaps.corpus.registry import build_source_registry, license_summary
from hcaps.corpus.sampling import generate_negative_pools
from hcaps.extraction.contracts import ExtractionTraceRecord, ProviderDisagreement
from hcaps.format.hashing import file_hash
from hcaps.format.manifest import AxpFileEntry
from hcaps.format.package import (
    load_manifest,
    write_manifest,
)
from hcaps.ingest.readers import read_source_documents
from hcaps.providers import ProviderCache, ProviderRequest, build_provider
from hcaps.providers.cache import ProviderCacheManifest
from hcaps.providers.cascade import ProviderCascade
from hcaps.providers.config import (
    ProviderEndpointConfig,
    ProviderIngressConfig,
    load_provider_ingress_config,
)
from hcaps.providers.traces import write_jsonl, write_trace_jsonl
from hcaps.substrate.builder import ClaimFamily, SubstrateBuildResult, build_substrate
from hcaps.substrate.manifest import SubstrateBuildConfig
from hcaps.utils.hashing import hash_record
from hcaps.utils.time import utc_now

CORPUS_BUILDER_VERSION = "0.1.0"


def build_corpus(config: CorpusBuildConfig) -> CorpusBuildResult:
    """Build a provider-aware claim-field corpus artifact set."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    provider_config = _provider_ingress_config(config)
    provider_config = _with_cache_mode(provider_config)
    provider_ids = _provider_ids(provider_config)
    substrate_config = _substrate_config(config)
    documents, ingest_warnings, _skipped = read_source_documents(
        substrate_config.input_path,
        substrate_config,
    )
    source_registry = build_source_registry(documents, provider_ids=provider_ids)

    substrate_result = build_substrate(substrate_config)
    cache = ProviderCache(provider_config.cache.root_path)
    cascade = _provider_cascade(provider_config, cache)
    provider_artifacts = _run_provider_tasks(substrate_result, cascade)
    negative_pools = generate_negative_pools(
        substrate_result.families,
        substrate_result.relations,
        provider_artifacts["disagreements"],
    )

    paths = _artifact_paths(output_dir)
    write_jsonl(
        paths["source_registry"],
        [entry.model_dump(mode="json") for entry in source_registry],
    )
    write_jsonl(paths["claim_families"], provider_artifacts["family_records"])
    write_jsonl(
        paths["relation_candidates"],
        [relation.model_dump(mode="json") for relation in substrate_result.relations],
    )
    write_jsonl(paths["negative_pools"], negative_pools)
    write_jsonl(paths["provider_disagreements"], provider_artifacts["disagreement_records"])
    write_trace_jsonl(paths["extraction_trace"], provider_artifacts["trace_records"])
    _write_json(
        paths["provider_cache_manifest"],
        cache.manifest().model_dump(mode="json"),
    )
    license_report = license_summary(source_registry)
    _write_json(paths["license_report"], license_report)
    leakage_report = _leakage_report(substrate_result)
    _write_json(paths["leakage_report"], leakage_report)
    quality_report = _quality_report(provider_artifacts["family_records"])
    _write_json(paths["quality_report"], quality_report)
    write_jsonl(paths["gold_candidates"], gold_candidate_records(paths["package"]))
    _write_context_stream(paths["package"], substrate_result)
    _augment_package(
        paths["package"],
        provider_cache_manifest=cache.manifest(),
        license_report=license_report,
        leakage_report=leakage_report,
        quality_report=quality_report,
    )

    artifact_hashes = _hash_artifacts(paths)
    warnings = [warning.message for warning in ingest_warnings]
    warnings.extend(warning.message for warning in substrate_result.manifest.warnings)
    corpus_manifest = CorpusManifest(
        corpus_id=config.corpus_id,
        corpus_version=config.corpus_version,
        created_at=utc_now(),
        domain=config.domain,
        source_count=len(source_registry),
        claim_family_count=len(substrate_result.families),
        capsule_count=len(substrate_result.axc_capsules),
        relation_candidate_count=len(substrate_result.relations),
        provider_policy=_provider_policy(provider_config),
        license_summary=license_report,
        temporal_policy={
            "cutoff_date": config.cutoff_date.isoformat() if config.cutoff_date else None,
            "strict_temporal_cutoff": config.strict_temporal_cutoff,
            "time_is_separate_from_lateral_context": True,
        },
        hashes=artifact_hashes,
        artifacts={key: str(path) for key, path in paths.items()},
        warnings=warnings,
    )
    _write_json(paths["corpus_manifest"], corpus_manifest.model_dump(mode="json"))
    return CorpusBuildResult(
        manifest=corpus_manifest,
        artifact_paths=paths,
        package_path=paths["package"],
    )


def _provider_ingress_config(config: CorpusBuildConfig) -> ProviderIngressConfig:
    if config.providers is not None:
        return config.providers
    if config.provider_config_path is not None:
        return load_provider_ingress_config(config.provider_config_path)
    return ProviderIngressConfig(
        primary=ProviderEndpointConfig(
            provider_id="deterministic",
            type="deterministic",
        )
    )


def _with_cache_mode(config: ProviderIngressConfig) -> ProviderIngressConfig:
    primary = config.primary.model_copy(update={"cache_mode": config.cache.mode})
    escalation = (
        config.escalation.model_copy(update={"cache_mode": config.cache.mode})
        if config.escalation is not None
        else None
    )
    return config.model_copy(update={"primary": primary, "escalation": escalation})


def _substrate_config(config: CorpusBuildConfig) -> SubstrateBuildConfig:
    output_dir = config.output_dir
    return SubstrateBuildConfig(
        input_path=config.input_path,
        output_path=output_dir / "capsules.jsonl",
        manifest_path=output_dir / "substrate_manifest.json",
        axc_output_path=output_dir / "capsules.axc",
        axp_package_path=output_dir / "dataset.axp",
        dataset_name=config.dataset_name,
        max_chunk_chars=config.max_chunk_chars,
        claim_family_similarity_threshold=config.claim_family_similarity_threshold,
        cutoff_date=config.cutoff_date,
        strict_temporal_cutoff=config.strict_temporal_cutoff,
        pdf_reader_enabled=config.pdf_reader_enabled,
        pdf_reader_required=config.pdf_reader_required,
    )


def _provider_cascade(config: ProviderIngressConfig, cache: ProviderCache) -> ProviderCascade:
    primary = build_provider(config.primary)
    escalation = build_provider(config.escalation) if config.escalation is not None else None
    return ProviderCascade(
        primary=primary,
        primary_config=config.primary,
        escalation=escalation,
        escalation_config=config.escalation,
        cache=cache,
        cascade_config=config.cascade,
    )


def _run_provider_tasks(
    substrate_result: SubstrateBuildResult,
    cascade: ProviderCascade,
) -> dict[str, Any]:
    trace_records: list[ExtractionTraceRecord] = []
    disagreement_records: list[dict[str, Any]] = []
    disagreements: list[ProviderDisagreement] = []
    family_records: list[dict[str, Any]] = []
    relation_degree = _relation_degree(substrate_result)
    provider_ids = [cascade.primary_config.provider_id]
    if cascade.escalation_config is not None:
        provider_ids.append(cascade.escalation_config.provider_id)

    for family in substrate_result.families:
        task_records: dict[str, dict[str, Any]] = {}
        for task in (
            "claim_extraction",
            "epistemic_extraction",
            "structured_view_generation",
        ):
            request = ProviderRequest(
                task=task,
                input_text=family.canonical_claim_text,
                source_ref={
                    "claim_family_id": family.family_id,
                    "source_document_ids": family.source_document_ids,
                    "domains": family.domains,
                },
                template_version=f"{task}_v0.1",
                context={
                    "source_count": family.independent_source_count_proxy,
                    "relation_degree": relation_degree.get(family.family_id, 0),
                    "provider_ids": provider_ids,
                },
            )
            run = cascade.run(request)
            trace = ExtractionTraceRecord(
                trace_id=f"trace_{hash_record({'family': family.family_id, 'task': task})[:24]}",
                created_at=utc_now(),
                provider_trace=run.response.provider_trace,
                task_input_hash=request.input_hash,
                normalized_output_hash=run.response.provider_trace.output_hash,
                warnings=run.response.warnings,
                metadata={
                    "claim_family_id": family.family_id,
                    "gate_decision": run.gate_decision.model_dump(mode="json"),
                    "merge_policy": run.merge.merge_policy,
                },
            )
            trace_records.append(trace)
            task_records[task] = {
                "trace_id": trace.trace_id,
                "output": run.response.normalized_output,
                "confidence": run.response.confidence,
                "provider_trace": run.response.provider_trace.model_dump(mode="json"),
            }
            for disagreement in run.merge.disagreements:
                disagreements.append(disagreement)
                disagreement_records.append(
                    {
                        "claim_family_id": family.family_id,
                        "task": task,
                        **disagreement.model_dump(mode="json"),
                    }
                )
        family_records.append(_family_record(family, task_records, relation_degree))
    return {
        "trace_records": trace_records,
        "disagreement_records": disagreement_records,
        "disagreements": disagreements,
        "family_records": family_records,
    }


def _family_record(
    family: ClaimFamily,
    task_records: dict[str, dict[str, Any]],
    relation_degree: dict[str, int],
) -> dict[str, Any]:
    view_output = task_records["structured_view_generation"]["output"]
    epistemic_output = task_records["epistemic_extraction"]["output"]
    confidence = float(task_records["claim_extraction"].get("confidence") or 0.0)
    disagreement_count = sum(
        1
        for record in task_records.values()
        if record["provider_trace"].get("merge_decision", "").startswith("selected_")
    )
    return {
        "family_id": family.family_id,
        "schema_claim_id": family.schema_claim_id,
        "canonical_claim_text": family.canonical_claim_text,
        "canonical_fingerprint": family.canonical_fingerprint,
        "member_claim_ids": family.member_claim_ids,
        "source_document_ids": family.source_document_ids,
        "domains": family.domains,
        "first_seen": family.first_seen.isoformat() if family.first_seen else None,
        "evidence_count": family.evidence_count,
        "independent_source_count_proxy": family.independent_source_count_proxy,
        "relation_degree": relation_degree.get(family.family_id, 0),
        "quality_tier": quality_tier(
            confidence=max(confidence, family.confidence_summary),
            warning_count=0,
            provider_disagreement_count=disagreement_count,
        ),
        "structured_views": view_output.get("views", {}),
        "epistemic_proxies": epistemic_output,
        "provider_trace_ids": {
            task: record["trace_id"] for task, record in sorted(task_records.items())
        },
        "provider_traces": {
            task: record["provider_trace"] for task, record in sorted(task_records.items())
        },
    }


def _relation_degree(result: SubstrateBuildResult) -> dict[str, int]:
    degree = {family.family_id: 0 for family in result.families}
    for relation in result.relations:
        degree[relation.source_family_id] = degree.get(relation.source_family_id, 0) + 1
        degree[relation.target_family_id] = degree.get(relation.target_family_id, 0) + 1
    return degree


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "corpus_manifest": output_dir / "corpus_manifest.json",
        "provider_cache_manifest": output_dir / "provider_cache_manifest.json",
        "extraction_trace": output_dir / "extraction_trace.jsonl",
        "provider_disagreements": output_dir / "provider_disagreements.jsonl",
        "source_registry": output_dir / "source_registry.jsonl",
        "claim_families": output_dir / "claim_families.jsonl",
        "relation_candidates": output_dir / "relation_candidates.jsonl",
        "negative_pools": output_dir / "negative_pools.jsonl",
        "gold_candidates": output_dir / "gold_candidates.jsonl",
        "leakage_report": output_dir / "leakage_report.json",
        "license_report": output_dir / "license_report.json",
        "quality_report": output_dir / "quality_report.json",
        "package": output_dir / "dataset.axp",
    }


def _leakage_report(result: SubstrateBuildResult) -> dict[str, Any]:
    warning_codes = [warning.code for warning in result.manifest.warnings]
    return {
        "temporal_leakage_detected": False,
        "cutoff": result.manifest.temporal_cutoff,
        "warning_codes": warning_codes,
        "notes": "Predictor-side source timestamps are checked by AXC/AXP validation.",
    }


def _quality_report(family_records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in family_records:
        tier = str(record["quality_tier"])
        counts[tier] = counts.get(tier, 0) + 1
    return {
        "quality_tier_counts": counts,
        "quality_tier_meaning": "construction confidence only",
    }


def _write_context_stream(package_path: Path, result: SubstrateBuildResult) -> None:
    records = []
    for capsule in result.axc_capsules:
        records.append(
            {
                "context_id": capsule.ids.context_id,
                "claim_family_id": capsule.ids.claim_family_id,
                "domains": capsule.context.domains,
                "communities": capsule.context.communities,
                "valid_as_of": capsule.temporal.valid_as_of.isoformat(),
                "provider_identity_is_lateral_context": True,
            }
        )
    write_jsonl(package_path / "data" / "contexts.axctx", records)


def _augment_package(
    package_path: Path,
    *,
    provider_cache_manifest: ProviderCacheManifest,
    license_report: dict[str, Any],
    leakage_report: dict[str, Any],
    quality_report: dict[str, Any],
) -> None:
    _write_json(
        package_path / "manifests" / "provider_report.json",
        provider_cache_manifest.model_dump(mode="json"),
    )
    _write_json(package_path / "manifests" / "license_report.json", license_report)
    _write_json(package_path / "manifests" / "leakage_report.json", leakage_report)
    _write_json(package_path / "manifests" / "quality_report.json", quality_report)
    manifest = load_manifest(package_path)
    extra_entries = [
        _file_entry(
            package_path,
            "data/contexts.axctx",
            "context_metadata",
            "application/x-ndjson",
            None,
        ),
        _file_entry(
            package_path,
            "manifests/provider_report.json",
            "provider_report",
            "application/json",
            None,
        ),
        _file_entry(
            package_path,
            "manifests/license_report.json",
            "license_report",
            "application/json",
            None,
        ),
        _file_entry(
            package_path,
            "manifests/quality_report.json",
            "quality_report",
            "application/json",
            None,
        ),
    ]
    files_by_path = {entry.relative_path: entry for entry in manifest.files}
    for entry in extra_entries:
        files_by_path[entry.relative_path] = entry
    files = [
        _file_entry(
            package_path,
            entry.relative_path,
            entry.role,
            entry.media_type,
            entry.record_count,
        )
        for entry in sorted(files_by_path.values(), key=lambda item: item.relative_path)
    ]
    hashes = {entry.relative_path: entry.sha256 for entry in files}
    _write_json(
        package_path / "manifests" / "files.json",
        [entry.model_dump(mode="json") for entry in files],
    )
    _write_json(package_path / "manifests" / "hashes.json", hashes)
    updated = manifest.model_copy(
        update={
            "files": files,
            "hashes": hashes,
            "counts": manifest.counts.model_copy(
                update={"contexts": _jsonl_count(package_path / "data" / "contexts.axctx")}
            ),
        }
    )
    write_manifest(package_path, updated)


def _file_entry(
    package_path: Path,
    relative_path: str,
    role: str,
    media_type: str,
    record_count: int | None,
) -> AxpFileEntry:
    path = package_path / relative_path
    return AxpFileEntry(
        relative_path=relative_path,
        role=role,
        media_type=media_type,
        byte_size=path.stat().st_size,
        sha256=file_hash(path),
        record_count=record_count,
    )


def _provider_ids(config: ProviderIngressConfig) -> list[str]:
    ids = [config.primary.provider_id]
    if config.escalation is not None:
        ids.append(config.escalation.provider_id)
    return ids


def _provider_policy(config: ProviderIngressConfig) -> dict[str, Any]:
    return {
        "primary": _provider_policy_entry(config.primary),
        "escalation": (
            _provider_policy_entry(config.escalation) if config.escalation is not None else None
        ),
        "cascade_enabled": config.cascade.enabled,
        "cache_root": str(config.cache.root_path),
        "cache_mode": config.cache.mode,
        "provider_use_scope": "data_construction_only",
    }


def _provider_policy_entry(config: ProviderEndpointConfig) -> dict[str, Any]:
    return {
        "provider_id": config.provider_id,
        "type": config.type,
        "provider_family": config.provider_family,
        "provider_mode": config.provider_mode,
        "model": config.model,
        "api_key_env_recorded": config.api_key_env,
        "dry_run": config.dry_run,
    }


def _hash_artifacts(paths: dict[str, Path]) -> dict[str, str]:
    hashes = {}
    for key, path in paths.items():
        if path.is_file():
            hashes[key] = file_hash(path)
    return hashes


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n"
    )


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_bytes().splitlines() if line.strip())
