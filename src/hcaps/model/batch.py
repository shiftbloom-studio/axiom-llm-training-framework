"""Batch normalization helpers for AXT-to-model handoff."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast


def records_from_batch(batch: object) -> list[dict[str, Any]]:
    """Return structured record dictionaries from supported P3 batch surfaces."""

    if hasattr(batch, "records"):
        records = cast(Any, batch).records
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            return [_normalize_record(record) for record in records]

    if hasattr(batch, "to_dict"):
        return records_from_batch(cast(Any, batch).to_dict())

    if isinstance(batch, Mapping):
        records = batch.get("records")
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            return [_normalize_record(record) for record in records]
        groups = batch.get("groups")
        if isinstance(groups, Mapping):
            return _records_from_group_mapping(groups)
        return [_normalize_record(batch)]

    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        return [_normalize_record(record) for record in batch]

    raise TypeError("expected AxtBatch, AxtRecord, mapping, or sequence of records")


def _normalize_record(record: object) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        value = cast(Any, record).to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError(f"unsupported AXT record type: {type(record).__name__}")


def _records_from_group_mapping(groups: Mapping[object, object]) -> list[dict[str, Any]]:
    group_items = {str(key): value for key, value in groups.items()}
    batch_size = 0
    for value in group_items.values():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            batch_size = max(batch_size, len(value))
    records: list[dict[str, Any]] = []
    for index in range(batch_size):
        record: dict[str, Any] = {}
        for group_name, values in group_items.items():
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                record[group_name] = values[index] if index < len(values) else {}
        records.append(record)
    return records
