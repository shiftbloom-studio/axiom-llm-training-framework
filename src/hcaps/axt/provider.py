"""Provider-context compilation for AXT."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .schema import MISSING_FLOAT, MISSING_INT, TensorGroup, as_f32, as_i64, stable_int64
from .vocab import VocabularyRegistry

SECRET_KEY_MARKERS = ("api_key", "apikey", "authorization", "bearer", "password", "secret", "token")


def compile_provider_context(
    capsules: list[Any],
    provider_traces: list[dict[str, object]],
    vocab: VocabularyRegistry,
    *,
    include_provider_context: bool,
    strict_secret_scan: bool = True,
) -> tuple[TensorGroup, dict[str, int]]:
    """Compile P1 provider traces into metadata/lateral-context tensors."""

    if strict_secret_scan:
        for trace_record in provider_traces:
            _assert_no_secret_payload(trace_record)

    trace_by_family = _traces_by_claim_family(provider_traces)
    provider_id: list[int] = []
    provider_family: list[int] = []
    provider_mode: list[int] = []
    provider_config_hash_ref: list[int] = []
    prompt_template: list[int] = []
    cascade_stage: list[int] = []
    escalation_reason: list[int] = []
    merge_strategy: list[int] = []
    merge_confidence: list[float] = []
    disagreement_score: list[float] = []
    cache_replay_flag: list[int] = []
    vote_distribution: list[list[float]] = []
    context_mask: list[int] = []
    active_count = 0

    for capsule in capsules:
        family_id = capsule.ids.claim_family_id
        trace: dict[str, object] | None = (
            trace_by_family.get(family_id) if include_provider_context else None
        )
        provider = _provider_payload(trace) if trace is not None else {}
        if provider:
            active_count += 1
        context_mask.append(int(bool(provider)))
        provider_id.append(vocab.lookup("provider_id", provider.get("provider_id")))
        provider_family.append(
            vocab.lookup(
                "provider_family", provider.get("provider_family") or provider.get("provider_type")
            )
        )
        provider_mode.append(vocab.lookup("provider_mode", provider.get("provider_mode")))
        provider_config_hash_ref.append(
            stable_int64(
                "provider_config",
                provider.get("provider_id"),
                provider.get("provider_model") or provider.get("model_name"),
            )
            if provider
            else MISSING_INT
        )
        prompt_template.append(
            vocab.lookup(
                "prompt_template",
                provider.get("prompt_template_version") or provider.get("prompt_template_id"),
            )
        )
        cascade_stage.append(vocab.lookup("cascade_stage", trace.get("task") if trace else None))
        escalation_reason.append(
            vocab.lookup("escalation_reason", provider.get("escalation_reason"))
        )
        merge_strategy.append(vocab.lookup("merge_strategy", provider.get("merge_decision")))
        confidence = provider.get("confidence")
        merge_confidence.append(
            float(confidence) if isinstance(confidence, int | float) else MISSING_FLOAT
        )
        disagreement_score.append(float(_disagreement_score(provider, trace)))
        cache_replay_flag.append(1 if provider.get("cache_key") else 0)
        vote_distribution.append(_vote_distribution(provider))

    group: TensorGroup = {
        "provider_id_idx": as_i64(provider_id),
        "provider_family_idx": as_i64(provider_family),
        "provider_mode_idx": as_i64(provider_mode),
        "provider_config_hash_ref": as_i64(provider_config_hash_ref),
        "prompt_template_idx": as_i64(prompt_template),
        "cascade_stage_idx": as_i64(cascade_stage),
        "escalation_reason_idx": as_i64(escalation_reason),
        "merge_strategy_idx": as_i64(merge_strategy),
        "merge_confidence": as_f32(merge_confidence),
        "disagreement_score": as_f32(disagreement_score),
        "provider_vote_distribution": np.asarray(vote_distribution, dtype=np.float32),
        "cache_replay_flag": as_i64(cache_replay_flag),
        "provider_context_mask": as_i64(context_mask),
    }
    return group, {
        "records_with_provider_context": active_count,
        "trace_records_loaded": len(provider_traces),
    }


def _traces_by_claim_family(
    provider_traces: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for trace in provider_traces:
        metadata = trace.get("metadata")
        if isinstance(metadata, dict):
            family_id = metadata.get("claim_family_id")
            if isinstance(family_id, str) and family_id not in indexed:
                indexed[family_id] = trace
    return indexed


def _provider_payload(trace: dict[str, object] | None) -> dict[str, object]:
    if trace is None:
        return {}
    provider = trace.get("provider_trace")
    return cast(dict[str, object], provider) if isinstance(provider, dict) else trace


def _disagreement_score(provider: dict[str, object], trace: dict[str, object] | None) -> float:
    score = provider.get("disagreement_score")
    if isinstance(score, int | float):
        return float(score)
    if provider.get("merge_decision") and str(provider["merge_decision"]).startswith("selected_"):
        return 1.0
    if trace is not None and trace.get("disagreement_type"):
        return 1.0
    return 0.0


def _vote_distribution(provider: dict[str, object]) -> list[float]:
    confidence = provider.get("confidence")
    selected = float(confidence) if isinstance(confidence, int | float) else 0.0
    disagreement = _disagreement_score(provider, None)
    abstain = max(0.0, 1.0 - selected)
    return [selected, disagreement, abstain]


def _assert_no_secret_payload(value: object, *, path: str = "provider_trace") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(marker in key_text for marker in SECRET_KEY_MARKERS):
                msg = f"secret-like provider trace key is forbidden in AXT: {path}.{key}"
                raise ValueError(msg)
            _assert_no_secret_payload(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_secret_payload(item, path=f"{path}[{index}]")
