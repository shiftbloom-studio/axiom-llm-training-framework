"""Deterministic hashing for AXF records and packages."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import orjson

STORED_HASH_KEYS = {"hash", "hashes", "sha256", "record_hash", "semantic_hash"}
GENERATED_ID_KEYS = {"capsule_id", "claim_state_id"}
NON_SEMANTIC_TIMESTAMP_KEYS = {"constructed_at", "created_at", "updated_at", "timestamp"}


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes with sorted keys."""

    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return sha256_bytes(payload.encode("utf-8"))


def file_hash(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_hash(record: Mapping[str, Any]) -> str:
    """Hash the full canonical record excluding stored hash fields only."""

    return sha256_bytes(canonical_json_bytes(_strip_keys(record, STORED_HASH_KEYS)))


def semantic_content_hash(record: Mapping[str, Any]) -> str:
    """Hash stable semantic content, excluding generated IDs and build timestamps."""

    excluded = STORED_HASH_KEYS | GENERATED_ID_KEYS | NON_SEMANTIC_TIMESTAMP_KEYS
    return sha256_bytes(canonical_json_bytes(_strip_keys(record, excluded)))


def _strip_keys(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _strip_keys(item_value, keys)
            for item_key, item_value in value.items()
            if item_key not in keys
        }
    if isinstance(value, list):
        return [_strip_keys(item, keys) for item in value]
    return value
