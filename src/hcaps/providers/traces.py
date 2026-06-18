"""Provider trace serialization helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson

from hcaps.extraction.contracts import ExtractionTraceRecord


def write_trace_jsonl(path: str | Path, records: Iterable[ExtractionTraceRecord]) -> int:
    """Write provider trace records as deterministic JSONL."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS))
            handle.write(b"\n")
            count += 1
    return count


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write plain JSONL artifact records."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record, option=orjson.OPT_SORT_KEYS))
            handle.write(b"\n")
            count += 1
    return count
