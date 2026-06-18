"""Deterministic hashing utilities for records and files."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import orjson


def canonical_json_bytes(record: Mapping[str, Any]) -> bytes:
    """Serialize a mapping in deterministic JSON form."""

    return orjson.dumps(record, option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE)


def hash_record(record: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for a JSON-compatible mapping."""

    return hashlib.sha256(canonical_json_bytes(record)).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hash of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
