"""Vocabulary registry parsing and stable categorical indexing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from hcaps.format.hashing import canonical_json_bytes, file_hash, sha256_bytes

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
DUAL_CATEGORY_COUNT = 2
PROVIDER_FAMILY_VALUES = {
    "deterministic",
    "openai_compatible",
    "python_callable",
    "human",
    "custom",
}
CASCADE_STAGE_VALUES = {"primary", "gate", "escalation", "merge", "fallback", "human_review"}


SECTION_TO_CATEGORY: dict[str, str] = {
    "Claim Types": "claim_type",
    "Relation Types": "relation_type",
    "Status Labels": "status_label",
    "Provider families": "provider_family",
    "Provider modes": "provider_mode",
    "Cascade stages": "cascade_stage",
    "Escalation reasons": "escalation_reason",
    "Context Fields": "context_field",
    "Negative Sample Types": "negative_sample_type",
    "Geometry Observable Names": "geometry_observable",
    "Text Projection Modes": "text_render_mode",
    "Special Tokens For Text Projection": "special_token",
}


class VocabularyRegistry(BaseModel):
    """Stable integer vocabularies used by one AXT bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    vocabularies: dict[str, dict[str, int]]

    def lookup(self, category: str, value: object | None, *, strict: bool = False) -> int:
        vocabulary = self.vocabularies.get(category)
        if vocabulary is None:
            if strict:
                msg = f"unknown vocabulary category: {category}"
                raise KeyError(msg)
            return 1
        normalized = _normalize_value(value)
        if normalized in vocabulary:
            return vocabulary[normalized]
        if strict:
            msg = f"unknown {category} value: {normalized}"
            raise KeyError(msg)
        return vocabulary[UNK_TOKEN]

    def stable_hash(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.model_dump(mode="json")))

    def to_json_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    def with_dynamic_values(
        self, values_by_category: Mapping[str, Iterable[object]]
    ) -> VocabularyRegistry:
        compiled = {category: dict(values) for category, values in self.vocabularies.items()}
        for category, values in values_by_category.items():
            vocabulary = compiled.setdefault(category, {PAD_TOKEN: 0, UNK_TOKEN: 1})
            for value in sorted({_normalize_value(item) for item in values if item is not None}):
                if value and value not in vocabulary:
                    vocabulary[value] = len(vocabulary)
        return self.model_copy(update={"vocabularies": compiled})


def load_vocabulary_registry(path: str | Path) -> VocabularyRegistry:
    registry_path = Path(path)
    vocabularies = _parse_registry(registry_path.read_text(encoding="utf-8"))
    _ensure_required_categories(vocabularies)
    return VocabularyRegistry(
        path=str(registry_path),
        sha256=file_hash(registry_path),
        vocabularies=vocabularies,
    )


def compile_vocabulary_registry(
    path: str | Path,
    *,
    dynamic_values: Mapping[str, Iterable[object]] | None = None,
) -> VocabularyRegistry:
    registry = load_vocabulary_registry(path)
    return registry.with_dynamic_values(dynamic_values or {})


def dynamic_values_from_records(
    capsules: Iterable[Any],
    provider_traces: Iterable[dict[str, object]],
    negative_pools: Iterable[dict[str, object]],
) -> dict[str, list[object]]:
    values: dict[str, list[object]] = {
        "claim_type": [],
        "relation_type": [],
        "status_label": [],
        "provider_id": [],
        "provider_family": [],
        "provider_mode": [],
        "cascade_stage": [],
        "escalation_reason": [],
        "merge_strategy": [],
        "prompt_template": [],
        "domain": [],
        "community": [],
        "method_context": [],
        "venue_context": [],
        "language": [],
        "register": [],
        "source_context": [],
        "source_license": [],
        "source_media_type": [],
        "negative_sample_type": [],
        "negative_sampling_method": [],
    }
    for capsule in capsules:
        claim = capsule.claim
        values["claim_type"].append(claim.claim_type)
        values["language"].append(claim.language)
        epistemic = capsule.epistemic_state
        values["status_label"].append(epistemic.status)
        context = capsule.context
        values["domain"].extend(context.domains)
        values["community"].extend(context.communities)
        provenance = capsule.provenance
        values["method_context"].append(provenance.construction_method)
        if provenance.extractor is not None:
            values["source_context"].append(provenance.extractor)
        for source in provenance.sources:
            values["source_license"].append(source.license)
            values["source_media_type"].append(_media_type_from_path(source.path))
        for relation in capsule.relations:
            values["relation_type"].append(relation.relation_type)
    for trace in provider_traces:
        provider = _provider_trace_payload(trace)
        values["provider_id"].append(provider.get("provider_id"))
        values["provider_family"].append(
            provider.get("provider_family") or provider.get("provider_type")
        )
        values["provider_mode"].append(provider.get("provider_mode"))
        values["cascade_stage"].append(trace.get("task") or provider.get("extraction_task"))
        values["escalation_reason"].append(provider.get("escalation_reason"))
        values["merge_strategy"].append(provider.get("merge_decision"))
        values["prompt_template"].append(provider.get("prompt_template_version"))
    for negative in negative_pools:
        values["negative_sample_type"].append(normalize_negative_type(negative))
        values["negative_sampling_method"].append(
            negative.get("sampling_method") or negative.get("pool_type")
        )
    return values


def normalize_negative_type(record: dict[str, object]) -> str:
    raw = str(record.get("negative_type") or record.get("pool_type") or "unknown")
    mapping = {
        "relation_hard_negative": "hard_relation_negative",
        "same_topic_unrelated": "same_context_unrelated",
        "near_but_distinct_claim": "near_claim_negative",
        "provenance_distractor_source": "provenance_negative",
        "temporal_distractor": "temporal_negative",
        "provider_disagreement_case": "provider_disagreement_negative",
    }
    return mapping.get(raw, raw)


def _parse_registry(text: str) -> dict[str, dict[str, int]]:
    vocabularies: dict[str, dict[str, int]] = {}
    active_categories: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            heading = line[3:].strip()
            if heading == "Provider Families And Modes":
                active_categories = ["provider_family", "provider_mode"]
            elif heading == "Cascade Stages And Escalation Reasons":
                active_categories = ["cascade_stage", "escalation_reason"]
            else:
                category = SECTION_TO_CATEGORY.get(heading)
                active_categories = [category] if category else []
            for category in active_categories:
                vocabularies.setdefault(category, {PAD_TOKEN: 0, UNK_TOKEN: 1})
            continue
        if not active_categories or not line.startswith("- `"):
            continue
        value = line.split("`", maxsplit=2)[1]
        category = (
            _dual_heading_category(active_categories, value)
            if len(active_categories) == DUAL_CATEGORY_COUNT
            else active_categories[0]
        )
        vocabulary = vocabularies.setdefault(category, {PAD_TOKEN: 0, UNK_TOKEN: 1})
        if value not in vocabulary:
            vocabulary[value] = len(vocabulary)
    return vocabularies


def _dual_heading_category(active_categories: list[str], value: str) -> str:
    first_category = active_categories[0]
    if first_category == "provider_family":
        return first_category if value in PROVIDER_FAMILY_VALUES else active_categories[1]
    if first_category == "cascade_stage":
        return first_category if value in CASCADE_STAGE_VALUES else active_categories[1]
    return first_category


def _ensure_required_categories(vocabularies: dict[str, dict[str, int]]) -> None:
    for category in (
        "claim_type",
        "relation_type",
        "status_label",
        "provider_family",
        "provider_mode",
        "cascade_stage",
        "escalation_reason",
        "negative_sample_type",
        "geometry_observable",
        "text_render_mode",
        "special_token",
    ):
        vocabularies.setdefault(category, {PAD_TOKEN: 0, UNK_TOKEN: 1})


def _normalize_value(value: object | None) -> str:
    if value is None:
        return UNK_TOKEN
    return str(value).strip() or UNK_TOKEN


def _provider_trace_payload(trace: dict[str, object]) -> dict[str, object]:
    payload = trace.get("provider_trace")
    return payload if isinstance(payload, dict) else trace


def _media_type_from_path(path: str | None) -> str:
    if not path:
        return "unknown"
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or "unknown"
