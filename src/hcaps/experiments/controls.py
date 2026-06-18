"""Deterministic P6 control transforms for experiment arms."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from hcaps.axt.batch import AxtBatch

CONTROL_TRANSFORMS: tuple[str, ...] = (
    "none",
    "context_shuffle",
    "provider_shuffle",
    "provenance_shuffle",
    "no_provenance",
    "no_relations",
    "no_context",
    "geometry_off",
    "popularity_frequency_control",
)

TEXT_MODES: tuple[str, ...] = ("flat_text", "structured_text", "capsule_text")
TransformHandler = Any


def _transform_handlers() -> dict[str, TransformHandler]:
    return {
        "context_shuffle": lambda records: _rotate_group(records, "lateral_context"),
        "provider_shuffle": lambda records: _rotate_group(records, "provider_context"),
        "provenance_shuffle": lambda records: _rotate_group(records, "provenance"),
        "no_provenance": lambda records: _clear_group(records, "provenance"),
        "no_relations": _clear_relation_groups,
        "no_context": lambda records: _clear_group(records, "lateral_context"),
        "popularity_frequency_control": _strip_frequency_like_records,
    }


def apply_control_transform(
    batch: AxtBatch,
    *,
    control_transform: str | None,
    text_mode: str | None,
) -> AxtBatch:
    """Return a deterministic transformed batch for a P6 arm.

    These transforms operate on predictor-side fields only. Target tensors and
    loss masks are preserved so missing labels never become negatives.
    """

    records = [deepcopy(record) for record in batch.records]
    if text_mode is not None:
        for record in records:
            _apply_text_mode(record, text_mode)
    transform = control_transform or "none"
    handler = _transform_handlers().get(transform)
    if handler is not None:
        handler(records)
    return AxtBatch(records=records)


def control_manifest(control_transform: str | None, text_mode: str | None) -> dict[str, Any]:
    return {
        "control_transform": control_transform or "none",
        "text_mode": text_mode or "structured_native",
        "provider_calls_allowed": False,
        "target_fields_modified": False,
        "future_target_inputs_added": False,
    }


def _apply_text_mode(record: dict[str, Any], text_mode: str) -> None:
    if text_mode not in TEXT_MODES:
        raise ValueError(f"unknown text projection mode: {text_mode}")
    text = record.get("text_projection")
    if not isinstance(text, dict):
        return
    prefix = text_mode
    input_ids = text.get(f"{prefix}_input_ids")
    attention_mask = text.get(f"{prefix}_attention_mask")
    if input_ids is not None:
        text["text_input_ids"] = input_ids
        text["text_labels"] = input_ids
        text["text_projection_targets"] = input_ids
    if attention_mask is not None:
        text["text_attention_mask"] = attention_mask
        text["text_loss_mask"] = attention_mask


def _rotate_group(records: list[dict[str, Any]], group_name: str) -> None:
    if len(records) <= 1:
        return
    values = [deepcopy(record.get(group_name, {})) for record in records]
    rotated = values[1:] + values[:1]
    for record, value in zip(records, rotated, strict=True):
        record[group_name] = value


def _clear_group(records: list[dict[str, Any]], group_name: str) -> None:
    for record in records:
        record[group_name] = {}


def _clear_relation_groups(records: list[dict[str, Any]]) -> None:
    for record in records:
        record["relations"] = {}
        record["relation_neighborhoods"] = {}


def _strip_frequency_like_records(records: list[dict[str, Any]]) -> None:
    for record in records:
        _strip_frequency_like_values(record)


def _strip_frequency_like_values(record: dict[str, Any]) -> None:
    for group_name in ("relations", "relation_neighborhoods", "epistemic_state"):
        group = record.get(group_name)
        if not isinstance(group, dict):
            continue
        for key in list(group):
            lowered = key.lower()
            if "degree" in lowered or "frequency" in lowered or "count" in lowered:
                value = group[key]
                if isinstance(value, list):
                    group[key] = [0 for _ in value]
                elif isinstance(value, int | float):
                    group[key] = 0
