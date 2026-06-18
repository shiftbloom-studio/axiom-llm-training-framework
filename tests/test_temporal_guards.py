from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest
from pydantic import ValidationError

from hcaps.schema.capsule import HoloCapsule

FIXTURES = Path(__file__).parent / "fixtures"


def _valid_record() -> dict[str, Any]:
    line = (FIXTURES / "valid_capsules.jsonl").read_bytes().splitlines()[0]
    record = orjson.loads(line)
    assert isinstance(record, dict)
    return record


def test_temporal_cutoff_guard_catches_future_input_leakage() -> None:
    record = _valid_record()
    record["provenance"][0]["source_timestamp"] = "1999-01-01T00:00:00Z"

    with pytest.raises(ValidationError, match="temporal leakage"):
        HoloCapsule.model_validate(record)


def test_temporal_cutoff_guard_catches_future_target_leakage() -> None:
    record = _valid_record()
    record["training_targets"]["target_timestamp"] = "1993-01-01T00:00:00Z"

    with pytest.raises(ValidationError, match="target leakage"):
        HoloCapsule.model_validate(record)
