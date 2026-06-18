"""AXC/AXP to AXT compiler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from hcaps.format.capsule import AxcCapsule
from hcaps.format.hashing import file_hash, sha256_bytes
from hcaps.format.package import CAPSULE_STREAM_PATH
from hcaps.format.streams import read_axc_stream

from .config import AxtCompileConfig
from .geometry import compile_geometry_slots
from .manifest import AxtManifest
from .masks import compile_availability_and_loss_masks
from .negatives import compile_negative_samples
from .neighborhoods import compile_relation_neighborhoods
from .provider import compile_provider_context
from .registry import FieldRegistry, load_field_registry
from .schema import (
    MISSING_FLOAT,
    MISSING_INT,
    REQUIRED_TENSOR_GROUPS,
    LoadedAxtInput,
    TensorGroup,
    TensorGroups,
    as_f32,
    as_i64,
    stable_int64,
    timestamp_seconds,
)
from .text_projection import compile_text_projection
from .vocab import VocabularyRegistry, compile_vocabulary_registry, dynamic_values_from_records
from .writer import write_axt_bundle


@dataclass(frozen=True)
class AxtCompileResult:
    """Result returned by the P3 compiler."""

    output_path: Path
    manifest: AxtManifest
    validation_report: dict[str, object]


def compile_axt(config: AxtCompileConfig, *, force: bool = False) -> AxtCompileResult:
    """Compile an AXC stream or AXP package into an AXT bundle."""

    loaded = load_axt_input(config)
    loaded = _normalize_loaded_side_family_ids(loaded)
    field_registry = load_field_registry(config.field_registry_path)
    vocabulary_registry = compile_vocabulary_registry(
        config.vocabulary_registry_path,
        dynamic_values=dynamic_values_from_records(
            loaded.capsules,
            loaded.provider_traces,
            loaded.negative_pools,
        ),
    )
    tensor_groups, source_index, summaries, reports, warnings = compile_tensor_groups(
        config=config,
        loaded=loaded,
        field_registry=field_registry,
        vocabulary_registry=vocabulary_registry,
    )
    missing_groups = sorted(set(REQUIRED_TENSOR_GROUPS) - set(tensor_groups))
    if missing_groups:
        msg = "AXT compiler did not produce required tensor groups: " + ", ".join(missing_groups)
        raise RuntimeError(msg)
    input_hash = _input_hash(loaded)
    manifest = write_axt_bundle(
        output_path=config.output_path,
        tensor_groups=tensor_groups,
        config=config,
        field_registry=field_registry,
        vocabulary_registry=vocabulary_registry,
        source_index=source_index,
        input_hash=input_hash,
        source_format=loaded.source_format,
        record_count=len(loaded.capsules),
        reports=reports,
        summaries=summaries,
        warnings=warnings,
        force=force,
    )
    validation_report = reports["validation_report"]
    return AxtCompileResult(
        output_path=config.output_path,
        manifest=manifest,
        validation_report=validation_report if isinstance(validation_report, dict) else {},
    )


def load_axt_input(config: AxtCompileConfig) -> LoadedAxtInput:
    """Load AXC/AXP records plus optional P1 side artifacts."""

    input_path = config.input_path
    if input_path.suffix == ".axc":
        capsules = list(read_axc_stream(input_path))
        return LoadedAxtInput(
            input_path=input_path,
            source_format="AXC",
            capsules_path=input_path,
            capsules=capsules,
            split_ids=None,
            provider_traces=_load_jsonl_candidates(
                input_path.parent, _provider_trace_candidates(input_path.parent)
            ),
            negative_pools=_load_jsonl_candidates(
                input_path.parent, _negative_pool_candidates(input_path.parent)
            ),
            evaluation_references=_load_jsonl_candidates(
                input_path.parent, _evaluation_candidates(input_path.parent)
            )
            if config.include_evaluation_references
            else [],
            source_registry=_load_jsonl_candidates(
                input_path.parent, _source_registry_candidates(input_path.parent)
            ),
            claim_families=_load_jsonl_candidates(
                input_path.parent, _claim_family_candidates(input_path.parent)
            ),
            package_reports={},
        )
    if input_path.suffix == ".axp" or input_path.is_dir():
        capsules_path = input_path / CAPSULE_STREAM_PATH
        if not capsules_path.exists():
            msg = f"AXP package missing canonical capsules stream: {capsules_path}"
            raise FileNotFoundError(msg)
        split_ids, splits = _load_split_ids(input_path, config)
        capsules = list(read_axc_stream(capsules_path))
        if split_ids is not None:
            capsules = [capsule for capsule in capsules if capsule.ids.capsule_id in split_ids]
        parent = input_path.parent
        return LoadedAxtInput(
            input_path=input_path,
            source_format="AXP",
            capsules_path=capsules_path,
            capsules=capsules,
            split_ids=split_ids,
            provider_traces=_load_jsonl_candidates(
                input_path, _provider_trace_candidates(input_path, parent)
            ),
            negative_pools=_load_jsonl_candidates(
                input_path, _negative_pool_candidates(input_path, parent)
            ),
            evaluation_references=_load_jsonl_candidates(
                input_path, _evaluation_candidates(input_path, parent)
            )
            if config.include_evaluation_references
            else [],
            source_registry=_load_jsonl_candidates(
                input_path, _source_registry_candidates(input_path, parent)
            ),
            claim_families=_load_jsonl_candidates(
                input_path, _claim_family_candidates(input_path, parent)
            ),
            package_reports={
                "splits": {name: sorted(ids) for name, ids in splits.items()},
                "reports": _load_package_reports(input_path),
            },
        )
    msg = f"unsupported AXT compiler input path: {input_path}"
    raise ValueError(msg)


def compile_tensor_groups(
    *,
    config: AxtCompileConfig,
    loaded: LoadedAxtInput,
    field_registry: FieldRegistry,
    vocabulary_registry: VocabularyRegistry,
) -> tuple[TensorGroups, dict[str, object], dict[str, object], dict[str, object], list[str]]:
    capsules = [AxcCapsule.model_validate(capsule) for capsule in loaded.capsules]
    lookup_maps = _lookup_maps(capsules, loaded.provider_traces)
    provider_group, provider_summary = compile_provider_context(
        capsules,
        loaded.provider_traces,
        vocabulary_registry,
        include_provider_context=config.include_provider_context,
        strict_secret_scan=config.strict_schema_validation,
    )
    negative_group, negative_summary = compile_negative_samples(
        capsules,
        loaded.negative_pools,
        lookup_maps["claim_family_to_index"],
        vocabulary_registry,
        include_negative_samples=config.include_negative_samples,
    )
    geometry_group, geometry_summary = compile_geometry_slots(
        capsules,
        include_geometry_slots=config.include_geometry_slots,
    )
    text_group, text_summary, text_sidecar = compile_text_projection(
        capsules,
        vocabulary_registry,
        render_modes=config.render_text_projection_modes,
        max_text_length=config.max_text_length,
    )
    provider_mask = provider_group["provider_context_mask"].astype(int).tolist()
    negative_mask = negative_group["negative_mask"].astype(int).tolist()
    availability_group, loss_group, availability_summary, loss_summary = (
        compile_availability_and_loss_masks(
            capsules,
            provider_context_mask=provider_mask,
            negative_sample_mask=negative_mask,
            evaluation_reference_count=len(loaded.evaluation_references),
        )
    )

    tensor_groups: TensorGroups = {
        "ids": _compile_id_tensors(capsules, loaded.provider_traces, lookup_maps),
        "claim": _compile_claim_tensors(capsules, lookup_maps, vocabulary_registry),
        "temporal": _compile_temporal_tensors(capsules, config),
        "lateral_context": _compile_lateral_context_tensors(capsules, vocabulary_registry),
        "provider_context": provider_group,
        "epistemic_state": _compile_epistemic_tensors(capsules, vocabulary_registry),
        "provenance": _compile_provenance_tensors(capsules, lookup_maps, vocabulary_registry),
        "relations": _compile_relation_tensors(capsules, lookup_maps, vocabulary_registry),
        "relation_neighborhoods": compile_relation_neighborhoods(
            capsules,
            lookup_maps["claim_family_to_index"],
            vocabulary_registry,
            include_relation_neighborhoods=config.include_relation_neighborhoods,
        ),
        "negative_samples": negative_group,
        "geometry_observables": geometry_group,
        "text_projection": text_group,
        "targets": _compile_target_tensors(capsules, lookup_maps, vocabulary_registry, text_group),
        "availability_masks": availability_group,
        "loss_masks": loss_group,
        "split_masks": _compile_split_masks(capsules, loaded.package_reports),
        "metadata": _compile_metadata_tensors(capsules, loaded),
    }
    source_index = _source_index(
        capsules,
        loaded=loaded,
        lookup_maps=lookup_maps,
        field_registry=field_registry,
        vocabulary_registry=vocabulary_registry,
        text_sidecar=text_sidecar,
    )
    mask_summary = {**availability_summary, **loss_summary}
    reports: dict[str, object] = {
        "validation_report": {
            "ok": True,
            "record_count": len(capsules),
            "source_format": loaded.source_format,
            "strict_schema_validation": config.strict_schema_validation,
            "time_is_separate_from_lateral_context": True,
            "text_projection_is_secondary": True,
            "issues": [],
        },
        "mask_report": {
            "availability": availability_summary,
            "loss": loss_summary,
            "missing_target_is_not_negative": True,
        },
        "missing_target_report": _missing_target_report(capsules, availability_group),
    }
    summaries: dict[str, object] = {
        "mask_summary": mask_summary,
        "negative_sample_summary": negative_summary,
        "provider_context_summary": provider_summary,
        "text_projection_summary": text_summary,
        "geometry_slot_summary": geometry_summary,
    }
    warnings = _compile_warnings(loaded, negative_summary, provider_summary)
    return tensor_groups, source_index, summaries, reports, warnings


def _compile_id_tensors(
    capsules: list[AxcCapsule],
    provider_traces: list[dict[str, object]],
    lookup_maps: dict[str, Any],
) -> TensorGroup:
    provider_trace_by_family = _provider_trace_index_by_family(provider_traces)
    source_to_index: dict[str, int] = lookup_maps["source_to_index"]
    return {
        "capsule_index": as_i64(list(range(len(capsules)))),
        "claim_family_index": as_i64(
            [
                lookup_maps["claim_family_to_index"][capsule.ids.claim_family_id]
                for capsule in capsules
            ]
        ),
        "claim_state_index": as_i64(
            [
                lookup_maps["claim_state_to_index"][capsule.ids.claim_state_id]
                for capsule in capsules
            ]
        ),
        "context_index": as_i64(
            [lookup_maps["context_to_index"][capsule.ids.context_id] for capsule in capsules]
        ),
        "source_index": as_i64(
            [
                source_to_index.get(capsule.provenance.sources[0].source_id, MISSING_INT)
                if capsule.provenance.sources
                else MISSING_INT
                for capsule in capsules
            ]
        ),
        "provider_trace_index": as_i64(
            [
                provider_trace_by_family.get(capsule.ids.claim_family_id, MISSING_INT)
                for capsule in capsules
            ]
        ),
    }


def _compile_claim_tensors(
    capsules: list[AxcCapsule],
    lookup_maps: dict[str, Any],
    vocab: VocabularyRegistry,
) -> TensorGroup:
    return {
        "claim_type_idx": as_i64(
            [vocab.lookup("claim_type", capsule.claim.claim_type) for capsule in capsules]
        ),
        "claim_family_idx": as_i64(
            [
                lookup_maps["claim_family_to_index"][capsule.ids.claim_family_id]
                for capsule in capsules
            ]
        ),
        "claim_state_idx": as_i64(
            [
                lookup_maps["claim_state_to_index"][capsule.ids.claim_state_id]
                for capsule in capsules
            ]
        ),
        "canonical_claim_text_ref": as_i64(
            [stable_int64(capsule.claim.canonical_text) for capsule in capsules]
        ),
        "claim_text_hash": as_i64(
            [stable_int64("claim_text", capsule.claim.canonical_text) for capsule in capsules]
        ),
        "claim_length_features": np.asarray(
            [
                [
                    len(capsule.claim.canonical_text.split()),
                    len(capsule.claim.canonical_text),
                    len(capsule.surface_forms.source_spans),
                ]
                for capsule in capsules
            ],
            dtype=np.int64,
        ),
    }


def _compile_temporal_tensors(capsules: list[AxcCapsule], config: AxtCompileConfig) -> TensorGroup:
    valid = [timestamp_seconds(capsule.temporal.valid_as_of) for capsule in capsules]
    observed = [timestamp_seconds(capsule.temporal.observed_at) for capsule in capsules]
    constructed = [timestamp_seconds(capsule.temporal.constructed_at) for capsule in capsules]
    source_publication = [
        timestamp_seconds(capsule.temporal.source_publication_date) for capsule in capsules
    ]
    retrieved = [MISSING_INT for _capsule in capsules]
    cutoff_mask: list[int] = []
    target_only_mask: list[int] = []
    future_target_mask: list[int] = []
    for capsule in capsules:
        target_only = any(source.target_only for source in capsule.provenance.sources) or any(
            span.target_only for span in capsule.surface_forms.source_spans
        )
        target_only_mask.append(int(target_only))
        future_target_mask.append(int(bool(capsule.training.future_label_fields)))
        cutoff_mask.append(
            int(
                capsule.temporal.leakage_validation_status == "passed"
                or not config.strict_temporal_masks
            )
        )
    return {
        "valid_as_of_timestamp": as_i64(valid),
        "observed_at_timestamp": as_i64(observed),
        "constructed_at_timestamp": as_i64(constructed),
        "source_publication_timestamp": as_i64(source_publication),
        "retrieved_at_timestamp": as_i64(retrieved),
        "time_delta_source_to_valid_as_of": as_i64(
            [
                _delta_or_missing(source, valid_at)
                for source, valid_at in zip(source_publication, valid, strict=True)
            ]
        ),
        "time_delta_observed_to_valid_as_of": as_i64(
            [
                _delta_or_missing(obs, valid_at)
                for obs, valid_at in zip(observed, valid, strict=True)
            ]
        ),
        "temporal_cutoff_mask": as_i64(cutoff_mask),
        "target_only_mask": as_i64(target_only_mask),
        "future_target_mask": as_i64(future_target_mask),
        "temporal_split_mask": as_i64([1] * len(capsules)),
    }


def _compile_lateral_context_tensors(
    capsules: list[AxcCapsule], vocab: VocabularyRegistry
) -> TensorGroup:
    return {
        "domain_idx": as_i64(
            [vocab.lookup("domain", _first(capsule.context.domains)) for capsule in capsules]
        ),
        "community_idx": as_i64(
            [vocab.lookup("community", _first(capsule.context.communities)) for capsule in capsules]
        ),
        "method_context_idx": as_i64(
            [
                vocab.lookup("method_context", capsule.provenance.construction_method)
                for capsule in capsules
            ]
        ),
        "venue_context_idx": as_i64([vocab.lookup("venue_context", None) for _capsule in capsules]),
        "language_idx": as_i64(
            [vocab.lookup("language", capsule.claim.language) for capsule in capsules]
        ),
        "register_idx": as_i64([vocab.lookup("register", None) for _capsule in capsules]),
        "source_context_idx": as_i64(
            [vocab.lookup("source_context", capsule.provenance.extractor) for capsule in capsules]
        ),
    }


def _compile_epistemic_tensors(
    capsules: list[AxcCapsule], vocab: VocabularyRegistry
) -> TensorGroup:
    proxy_names = (
        "ontology_compatibility",
        "evidential_anchoring",
        "transformation_pressure",
        "independent_redundancy",
        "uncertainty",
    )
    values: dict[str, list[float]] = {f"{name}_value": [] for name in proxy_names}
    confidences: dict[str, list[float]] = {f"{name}_confidence": [] for name in proxy_names}
    method_idx: list[list[int]] = []
    basis_hash: list[list[int]] = []
    status_idx: list[int] = []
    for capsule in capsules:
        epistemic = capsule.epistemic_state
        status_idx.append(vocab.lookup("status_label", epistemic.status))
        row_methods: list[int] = []
        row_hashes: list[int] = []
        for name in proxy_names:
            proxy = getattr(epistemic, name)
            if name == "independent_redundancy":
                values[f"{name}_value"].append(float(proxy.effective_count))
                confidences[f"{name}_confidence"].append(
                    float(proxy.confidence) if proxy.confidence is not None else MISSING_FLOAT
                )
                method = str(proxy.method)
                basis = proxy.notes or method
            elif proxy is not None:
                values[f"{name}_value"].append(float(proxy.value))
                confidences[f"{name}_confidence"].append(
                    float(proxy.confidence) if proxy.confidence is not None else MISSING_FLOAT
                )
                method = proxy.method
                basis = method
            else:
                values[f"{name}_value"].append(MISSING_FLOAT)
                confidences[f"{name}_confidence"].append(MISSING_FLOAT)
                method = None
                basis = None
            row_methods.append(vocab.lookup("method_context", method))
            row_hashes.append(stable_int64("proxy_basis", basis) if basis else MISSING_INT)
        method_idx.append(row_methods)
        basis_hash.append(row_hashes)
    group: TensorGroup = {
        "status_idx": as_i64(status_idx),
        "proxy_method_idx": np.asarray(method_idx, dtype=np.int64),
        "proxy_basis_hash": np.asarray(basis_hash, dtype=np.int64),
    }
    group.update({name: as_f32(items) for name, items in values.items()})
    group.update({name: as_f32(items) for name, items in confidences.items()})
    return group


def _compile_provenance_tensors(
    capsules: list[AxcCapsule],
    lookup_maps: dict[str, Any],
    vocab: VocabularyRegistry,
) -> TensorGroup:
    source_values: list[int] = []
    source_dates: list[int] = []
    license_idx: list[int] = []
    media_type_idx: list[int] = []
    source_hash_refs: list[int] = []
    span_refs: list[int] = []
    source_offsets = [0]
    span_offsets = [0]
    source_to_index: dict[str, int] = lookup_maps["source_to_index"]
    for capsule in capsules:
        for source in capsule.provenance.sources:
            source_values.append(source_to_index[source.source_id])
            source_dates.append(timestamp_seconds(source.source_date))
            license_idx.append(vocab.lookup("source_license", source.license))
            media_type_idx.append(
                vocab.lookup(
                    "source_media_type", Path(source.path or "").suffix.lstrip(".") or "unknown"
                )
            )
            source_hash_refs.append(stable_int64("source_hash", source.sha256 or source.source_id))
        for span in capsule.surface_forms.source_spans:
            span_refs.append(
                stable_int64("span", span.span_id, span.source_id, span.start_char, span.end_char)
            )
        source_offsets.append(len(source_values))
        span_offsets.append(len(span_refs))
    source_mask = [
        1 if source_offsets[index + 1] > source_offsets[index] else 0
        for index in range(len(capsules))
    ]
    return {
        "provenance_source_values": as_i64(source_values),
        "provenance_source_offsets": as_i64(source_offsets),
        "provenance_source_mask": as_i64(source_mask),
        "source_date_values": as_i64(source_dates),
        "source_license_idx": as_i64(license_idx),
        "source_media_type_idx": as_i64(media_type_idx),
        "source_hash_refs": as_i64(source_hash_refs),
        "span_refs": as_i64(span_refs),
        "span_offsets": as_i64(span_offsets),
    }


def _compile_relation_tensors(
    capsules: list[AxcCapsule],
    lookup_maps: dict[str, Any],
    vocab: VocabularyRegistry,
) -> TensorGroup:
    relation_type_values: list[int] = []
    target_values: list[int] = []
    confidence_values: list[float] = []
    evidence_span_refs: list[int] = []
    offsets = [0]
    family_to_index: dict[str, int] = lookup_maps["claim_family_to_index"]
    for capsule in capsules:
        for relation in capsule.relations:
            relation_type_values.append(vocab.lookup("relation_type", relation.relation_type))
            target_values.append(family_to_index.get(relation.target_claim_family_id, MISSING_INT))
            confidence_values.append(
                float(relation.confidence) if relation.confidence is not None else MISSING_FLOAT
            )
            evidence_span_refs.append(
                stable_int64("relation_evidence", *relation.evidence_span_ids)
            )
        offsets.append(len(relation_type_values))
    mask = [1 if offsets[index + 1] > offsets[index] else 0 for index in range(len(capsules))]
    return {
        "relation_type_values": as_i64(relation_type_values),
        "relation_target_claim_family_values": as_i64(target_values),
        "relation_confidence_values": as_f32(confidence_values),
        "relation_offsets": as_i64(offsets),
        "relation_mask": as_i64(mask),
        "relation_evidence_span_refs": as_i64(evidence_span_refs),
    }


def _compile_target_tensors(
    capsules: list[AxcCapsule],
    lookup_maps: dict[str, Any],
    vocab: VocabularyRegistry,
    text_group: TensorGroup,
) -> TensorGroup:
    relation_types: list[int] = []
    relation_ids: list[int] = []
    relation_offsets = [0]
    provenance_ids: list[int] = []
    provenance_offsets = [0]
    family_to_index: dict[str, int] = lookup_maps["claim_family_to_index"]
    source_to_index: dict[str, int] = lookup_maps["source_to_index"]
    for capsule in capsules:
        for relation in capsule.relations:
            relation_types.append(vocab.lookup("relation_type", relation.relation_type))
            relation_ids.append(family_to_index.get(relation.target_claim_family_id, MISSING_INT))
        relation_offsets.append(len(relation_types))
        for source in capsule.provenance.sources:
            provenance_ids.append(source_to_index.get(source.source_id, MISSING_INT))
        provenance_offsets.append(len(provenance_ids))
    return {
        "relation_target_types": as_i64(relation_types),
        "relation_target_ids": as_i64(relation_ids),
        "relation_target_offsets": as_i64(relation_offsets),
        "provenance_target_ids": as_i64(provenance_ids),
        "provenance_target_offsets": as_i64(provenance_offsets),
        "status_target_idx": as_i64(
            [vocab.lookup("status_label", capsule.epistemic_state.status) for capsule in capsules]
        ),
        "uncertainty_target": as_f32(
            [
                float(capsule.epistemic_state.uncertainty.value)
                if capsule.epistemic_state.uncertainty is not None
                else MISSING_FLOAT
                for capsule in capsules
            ]
        ),
        "redundancy_target": as_f32(
            [
                float(capsule.epistemic_state.independent_redundancy.effective_count)
                for capsule in capsules
            ]
        ),
        "future_summary_target_ref": as_i64(
            [
                stable_int64("future_summary", capsule.ids.capsule_id)
                if capsule.training.future_label_fields
                else MISSING_INT
                for capsule in capsules
            ]
        ),
        "geometry_observable_targets": np.asarray(
            [
                [
                    float(capsule.geometry.curvature_score)
                    if capsule.geometry.curvature_score is not None
                    else MISSING_FLOAT,
                    float(
                        capsule.geometry.transport_observables.get("holonomy_norm", MISSING_FLOAT)
                    ),
                    float(
                        capsule.geometry.transport_observables.get("trace_summary", MISSING_FLOAT)
                    ),
                    float(
                        capsule.geometry.transport_observables.get(
                            "spectrum_summary", MISSING_FLOAT
                        )
                    ),
                    float(
                        capsule.geometry.transport_observables.get(
                            "context_lability", MISSING_FLOAT
                        )
                    ),
                ]
                for capsule in capsules
            ],
            dtype=np.float32,
        ),
        "text_projection_targets": text_group["text_projection_targets"],
        "target_axc_out_layer_idx": as_i64(
            [stable_int64("validated_axc_out") for _capsule in capsules]
        ),
    }


def _compile_split_masks(
    capsules: list[AxcCapsule], package_reports: dict[str, object]
) -> TensorGroup:
    split_payload = package_reports.get("splits", {})
    splits = split_payload if isinstance(split_payload, dict) else {}
    group: TensorGroup = {"selected_split_mask": as_i64([1] * len(capsules))}
    capsule_ids = [capsule.ids.capsule_id for capsule in capsules]
    for split_name, values in splits.items():
        ids = set(values) if isinstance(values, list) else set()
        group[f"{split_name}_mask"] = as_i64(
            [1 if capsule_id in ids else 0 for capsule_id in capsule_ids]
        )
    if not splits:
        group["all_mask"] = as_i64([1] * len(capsules))
    return group


def _compile_metadata_tensors(capsules: list[AxcCapsule], loaded: LoadedAxtInput) -> TensorGroup:
    source_format_idx = 1 if loaded.source_format == "AXP" else 0
    return {
        "record_index": as_i64(list(range(len(capsules)))),
        "source_format_idx": as_i64([source_format_idx] * len(capsules)),
        "input_record_hash": as_i64(
            [stable_int64(capsule.model_dump(mode="json")) for capsule in capsules]
        ),
        "evaluation_reference_count": as_i64([len(loaded.evaluation_references)] * len(capsules)),
    }


def _lookup_maps(
    capsules: list[AxcCapsule], provider_traces: list[dict[str, object]]
) -> dict[str, Any]:
    claim_family_ids = _unique([capsule.ids.claim_family_id for capsule in capsules])
    claim_state_ids = _unique([capsule.ids.claim_state_id for capsule in capsules])
    context_ids = _unique([capsule.ids.context_id for capsule in capsules])
    source_ids = _unique(
        [source.source_id for capsule in capsules for source in capsule.provenance.sources]
    )
    provider_trace_ids = _unique(
        [trace_id for trace in provider_traces if (trace_id := _trace_id(trace)) is not None]
    )
    return {
        "index_to_capsule_id": [capsule.ids.capsule_id for capsule in capsules],
        "index_to_claim_family_id": claim_family_ids,
        "index_to_claim_state_id": claim_state_ids,
        "index_to_context_id": context_ids,
        "index_to_source_id": source_ids,
        "index_to_provider_trace_id": provider_trace_ids,
        "claim_family_to_index": {value: index for index, value in enumerate(claim_family_ids)},
        "claim_state_to_index": {value: index for index, value in enumerate(claim_state_ids)},
        "context_to_index": {value: index for index, value in enumerate(context_ids)},
        "source_to_index": {value: index for index, value in enumerate(source_ids)},
        "provider_trace_to_index": {value: index for index, value in enumerate(provider_trace_ids)},
    }


def _source_index(
    capsules: list[AxcCapsule],
    *,
    loaded: LoadedAxtInput,
    lookup_maps: dict[str, Any],
    field_registry: FieldRegistry,
    vocabulary_registry: VocabularyRegistry,
    text_sidecar: dict[str, object],
) -> dict[str, object]:
    return {
        "input_path": str(loaded.input_path),
        "source_format": loaded.source_format,
        "capsules_path": str(loaded.capsules_path),
        "lookup_maps": {
            key: value for key, value in lookup_maps.items() if key.startswith("index_to_")
        },
        "records": [
            {
                "capsule_id": capsule.ids.capsule_id,
                "claim_family_id": capsule.ids.claim_family_id,
                "claim_state_id": capsule.ids.claim_state_id,
                "context_id": capsule.ids.context_id,
            }
            for capsule in capsules
        ],
        "field_visibility": {
            entry.field_name: entry.visibility for entry in field_registry.entries
        },
        "vocabularies": vocabulary_registry.vocabularies,
        "text_projection": text_sidecar,
        "provider_trace_count": len(loaded.provider_traces),
        "negative_pool_count": len(loaded.negative_pools),
        "evaluation_reference_count": len(loaded.evaluation_references),
        "claim_family_side_record_count": len(loaded.claim_families),
    }


def _missing_target_report(
    capsules: list[AxcCapsule], availability_group: TensorGroup
) -> dict[str, object]:
    missing: list[dict[str, object]] = []
    for index, capsule in enumerate(capsules):
        if int(availability_group["available_relation_targets"][index]) == 0:
            missing.append(
                {
                    "capsule_id": capsule.ids.capsule_id,
                    "target": "relation",
                    "state": "missing",
                    "loss_active": False,
                }
            )
        if int(availability_group["available_future_targets"][index]) == 0:
            missing.append(
                {
                    "capsule_id": capsule.ids.capsule_id,
                    "target": "future_summary",
                    "state": "missing",
                    "loss_active": False,
                }
            )
    return {"missing_targets": missing, "missing_target_is_not_negative": True}


def _compile_warnings(
    loaded: LoadedAxtInput,
    negative_summary: dict[str, int],
    provider_summary: dict[str, int],
) -> list[str]:
    warnings: list[str] = []
    if provider_summary.get("trace_records_loaded", 0) == 0:
        warnings.append(
            "No provider traces were available; provider context tensors use <unk>/missing values."
        )
    if negative_summary.get("negative_records_compiled", 0) == 0:
        warnings.append("No negative pools were available for selected records.")
    if loaded.source_format == "AXP" and loaded.split_ids is None:
        warnings.append("AXP compiled without a split because allow_all_without_split=true.")
    return warnings


def _load_split_ids(
    package_path: Path,
    config: AxtCompileConfig,
) -> tuple[set[str] | None, dict[str, set[str]]]:
    splits_dir = package_path / "splits"
    splits: dict[str, set[str]] = {}
    for split_file in sorted(splits_dir.glob("*.json")):
        splits[split_file.stem] = _read_split_ids(split_file)
    if config.split_name is None:
        if config.allow_all_without_split:
            return None, splits
        msg = "AXP compilation requires split_name or allow_all_without_split=true"
        raise ValueError(msg)
    split_path = splits_dir / f"{config.split_name}.json"
    if not split_path.exists():
        if config.allow_all_without_split:
            return None, splits
        msg = f"AXP split is missing: {split_path}"
        raise FileNotFoundError(msg)
    return _read_split_ids(split_path), splits


def _read_split_ids(path: Path) -> set[str]:
    payload = orjson.loads(path.read_bytes())
    value = payload.get("capsule_ids", []) if isinstance(payload, dict) else payload
    if not isinstance(value, list):
        msg = f"split file must contain a capsule_ids list: {path}"
        raise ValueError(msg)
    return {item for item in value if isinstance(item, str)}


def _load_jsonl_candidates(base: Path, paths: list[Path]) -> list[dict[str, object]]:
    del base
    records: list[dict[str, object]] = []
    seen_paths: set[Path] = set()
    for path in paths:
        if path in seen_paths or not path.exists() or not path.is_file():
            continue
        seen_paths.add(path)
        records.extend(_read_jsonl(path))
    return records


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        payload = orjson.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _load_package_reports(package_path: Path) -> dict[str, object]:
    reports: dict[str, object] = {}
    for directory in ("manifests", "reports"):
        report_dir = package_path / directory
        if not report_dir.exists():
            continue
        for path in sorted(report_dir.glob("*.json")):
            try:
                reports[path.stem] = orjson.loads(path.read_bytes())
            except orjson.JSONDecodeError:
                continue
    return reports


def _provider_trace_candidates(primary: Path, secondary: Path | None = None) -> list[Path]:
    roots = [primary]
    if secondary is not None:
        roots.append(secondary)
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "extraction_trace.jsonl",
                root / "provider_traces.jsonl",
                root / "data" / "provider_traces.axprovtrace",
                root / "data" / "extraction_trace.jsonl",
                root / "manifests" / "provider_traces.jsonl",
                root / "reports" / "provider_traces.axprovtrace",
            ]
        )
    return candidates


def _negative_pool_candidates(primary: Path, secondary: Path | None = None) -> list[Path]:
    roots = [primary]
    if secondary is not None:
        roots.append(secondary)
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "negative_pools.jsonl",
                root / "data" / "negative_pools.axneg",
                root / "data" / "negative_pools.jsonl",
                root / "manifests" / "negative_pools.jsonl",
            ]
        )
    return candidates


def _evaluation_candidates(primary: Path, secondary: Path | None = None) -> list[Path]:
    roots = [primary]
    if secondary is not None:
        roots.append(secondary)
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "gold_candidates.jsonl",
                root / "evaluation_references.jsonl",
                root / "data" / "evaluation_references.axeval",
            ]
        )
    return candidates


def _claim_family_candidates(primary: Path, secondary: Path | None = None) -> list[Path]:
    roots = [primary]
    if secondary is not None:
        roots.append(secondary)
    candidates: list[Path] = []
    for root in roots:
        candidates.extend([root / "claim_families.jsonl", root / "data" / "claim_families.axcfam"])
    return candidates


def _source_registry_candidates(primary: Path, secondary: Path | None = None) -> list[Path]:
    roots = [primary]
    if secondary is not None:
        roots.append(secondary)
    candidates: list[Path] = []
    for root in roots:
        candidates.extend([root / "source_registry.jsonl", root / "data" / "sources.axsrc"])
    return candidates


def _normalize_loaded_side_family_ids(loaded: LoadedAxtInput) -> LoadedAxtInput:
    capsules = [AxcCapsule.model_validate(capsule) for capsule in loaded.capsules]
    aliases = _family_aliases(capsules, loaded.claim_families)
    if not aliases:
        return loaded
    return LoadedAxtInput(
        input_path=loaded.input_path,
        source_format=loaded.source_format,
        capsules_path=loaded.capsules_path,
        capsules=loaded.capsules,
        split_ids=loaded.split_ids,
        provider_traces=[
            _rewrite_family_refs(record, aliases) for record in loaded.provider_traces
        ],
        negative_pools=[_rewrite_family_refs(record, aliases) for record in loaded.negative_pools],
        evaluation_references=[
            _rewrite_family_refs(record, aliases) for record in loaded.evaluation_references
        ],
        source_registry=loaded.source_registry,
        claim_families=loaded.claim_families,
        package_reports=loaded.package_reports,
    )


def _family_aliases(
    capsules: list[AxcCapsule],
    claim_families: list[dict[str, object]],
) -> dict[str, str]:
    axc_by_text = {
        capsule.claim.canonical_text.strip(): capsule.ids.claim_family_id for capsule in capsules
    }
    aliases: dict[str, str] = {}
    for record in claim_families:
        family_id = record.get("family_id")
        canonical = record.get("canonical_claim_text") or record.get("canonical_text")
        if isinstance(family_id, str) and isinstance(canonical, str):
            axc_id = axc_by_text.get(canonical.strip())
            if axc_id is not None:
                aliases[family_id] = axc_id
    return aliases


def _rewrite_family_refs(value: object, aliases: dict[str, str]) -> Any:
    if isinstance(value, dict):
        rewritten: dict[str, object] = {}
        for key, item in value.items():
            if key in {
                "claim_family_id",
                "source_family_id",
                "target_family_id",
                "source_claim_family_id",
                "target_claim_family_id",
                "distractor_family_id",
                "later_family_id",
            } and isinstance(item, str):
                rewritten[key] = aliases.get(item, item)
            else:
                rewritten[key] = _rewrite_family_refs(item, aliases)
        return rewritten
    if isinstance(value, list):
        return [_rewrite_family_refs(item, aliases) for item in value]
    return value


def _input_hash(loaded: LoadedAxtInput) -> str:
    if loaded.source_format == "AXC":
        return file_hash(loaded.capsules_path)
    hashes: list[str] = []
    for path in sorted(loaded.input_path.rglob("*")):
        if path.is_file():
            hashes.append(f"{path.relative_to(loaded.input_path).as_posix()}:{file_hash(path)}")
    return sha256_bytes("\n".join(hashes).encode("utf-8"))


def _provider_trace_index_by_family(provider_traces: list[dict[str, object]]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for index, trace in enumerate(provider_traces):
        metadata = trace.get("metadata")
        if isinstance(metadata, dict):
            family_id = metadata.get("claim_family_id")
            if isinstance(family_id, str) and family_id not in lookup:
                lookup[family_id] = index
    return lookup


def _trace_id(trace: dict[str, object]) -> str | None:
    value = trace.get("trace_id") or trace.get("provider_trace_id")
    return value if isinstance(value, str) else None


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _delta_or_missing(left: int, right: int) -> int:
    if MISSING_INT in (left, right):
        return MISSING_INT
    return right - left


def _first(values: list[str]) -> str | None:
    return values[0] if values else None
