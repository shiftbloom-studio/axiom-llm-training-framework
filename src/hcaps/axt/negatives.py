"""Negative-pool normalization and tensor compilation."""

from __future__ import annotations

from typing import Any

from .schema import MISSING_FLOAT, MISSING_INT, TensorGroup, as_f32, as_i64, stable_int64
from .vocab import VocabularyRegistry, normalize_negative_type

LOSS_ELIGIBLE_TYPES = {
    "near_topic_negative",
    "near_claim_negative",
    "same_context_unrelated",
    "same_source_unrelated",
    "contradiction_candidate",
    "supersession_candidate",
    "temporal_negative",
    "provider_disagreement_negative",
    "hard_relation_negative",
    "provenance_negative",
}


def compile_negative_samples(
    capsules: list[Any],
    negative_pools: list[dict[str, object]],
    family_to_index: dict[str, int],
    vocab: VocabularyRegistry,
    *,
    include_negative_samples: bool,
) -> tuple[TensorGroup, dict[str, int]]:
    """Compile typed negative samples into ragged AXT tensors."""

    records_by_family = _records_by_family(negative_pools) if include_negative_samples else {}
    type_values: list[int] = []
    source_values: list[int] = []
    target_values: list[int] = []
    hardness_values: list[float] = []
    method_values: list[int] = []
    eligibility: list[int] = []
    offsets = [0]
    active_rows = 0

    for capsule in capsules:
        family_id = capsule.ids.claim_family_id
        records = records_by_family.get(family_id, [])
        for record in records:
            negative_type = normalize_negative_type(record)
            source_family = _source_family(record) or family_id
            target_family = _target_family(record)
            type_values.append(vocab.lookup("negative_sample_type", negative_type))
            source_values.append(family_to_index.get(source_family, MISSING_INT))
            target_values.append(family_to_index.get(target_family or "", MISSING_INT))
            hardness_values.append(_hardness(record))
            method_values.append(
                vocab.lookup(
                    "negative_sampling_method",
                    record.get("sampling_method") or record.get("pool_type"),
                )
            )
            eligibility.append(1 if negative_type in LOSS_ELIGIBLE_TYPES else 0)
            active_rows += 1
        offsets.append(len(type_values))

    masks = [1 if offsets[index + 1] > offsets[index] else 0 for index in range(len(capsules))]
    group: TensorGroup = {
        "negative_type_idx": as_i64(type_values),
        "negative_source_idx": as_i64(source_values),
        "negative_target_idx": as_i64(target_values),
        "negative_hardness": as_f32(hardness_values),
        "negative_sampling_method_idx": as_i64(method_values),
        "negative_offsets": as_i64(offsets),
        "negative_mask": as_i64(masks),
        "negative_loss_eligibility_mask": as_i64(eligibility),
    }
    return group, {
        "negative_records_loaded": len(negative_pools),
        "negative_records_compiled": active_rows,
        "records_with_negatives": sum(masks),
    }


def _records_by_family(records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    indexed: dict[str, list[dict[str, object]]] = {}
    for record in records:
        family_id = _source_family(record)
        if family_id is None:
            continue
        indexed.setdefault(family_id, []).append(record)
    return indexed


def _source_family(record: dict[str, object]) -> str | None:
    value = record.get("source_claim_family_id") or record.get("source_family_id")
    return value if isinstance(value, str) else None


def _target_family(record: dict[str, object]) -> str | None:
    for key in (
        "target_claim_family_id",
        "target_family_id",
        "distractor_family_id",
        "later_family_id",
        "target_id",
    ):
        value = record.get(key)
        if isinstance(value, str):
            return value
    target_hash_seed = record.get("remote_output_hash") or record.get("local_output_hash")
    return (
        f"provider_disagreement:{target_hash_seed}" if isinstance(target_hash_seed, str) else None
    )


def _hardness(record: dict[str, object]) -> float:
    for key in ("hardness_score", "score", "similarity"):
        value = record.get(key)
        if isinstance(value, int | float):
            return float(value)
    if record.get("pool_type") == "provider_disagreement_case":
        return 1.0
    return MISSING_FLOAT


def negative_record_ref(record: dict[str, object]) -> int:
    return stable_int64("negative", record.get("negative_id") or record.get("pool_id") or record)
