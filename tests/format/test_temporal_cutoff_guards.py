from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from hcaps.format.capsule import AxcCapsule


def test_post_valid_as_of_source_fails_unless_target_only(minimal_axc_record: Any) -> None:
    record = deepcopy(minimal_axc_record)
    record["provenance"]["sources"][0]["source_date"] = "2025-01-01T00:00:00Z"

    with pytest.raises(ValidationError, match="temporal leakage"):
        AxcCapsule.model_validate(record)

    record["provenance"]["sources"][0]["target_only"] = True
    assert AxcCapsule.model_validate(record)


def test_source_date_equal_to_valid_as_of_passes(minimal_axc_record: Any) -> None:
    record = deepcopy(minimal_axc_record)
    record["provenance"]["sources"][0]["source_date"] = "2024-01-01T00:00:00Z"
    record["surface_forms"]["source_spans"][0]["source_date"] = "2024-01-01T00:00:00Z"
    record["temporal"]["source_publication_date"] = "2024-01-01T00:00:00Z"

    assert AxcCapsule.model_validate(record)


def test_constructed_at_after_valid_as_of_is_allowed(minimal_axc_record: Any) -> None:
    record = deepcopy(minimal_axc_record)
    record["temporal"]["constructed_at"] = "2026-06-18T00:00:00Z"

    assert AxcCapsule.model_validate(record)


def test_future_source_span_passes_when_marked_target_only(minimal_axc_record: Any) -> None:
    record = deepcopy(minimal_axc_record)
    record["surface_forms"]["source_spans"][0]["source_date"] = "2025-01-01T00:00:00Z"
    record["surface_forms"]["source_spans"][0]["target_only"] = True

    assert AxcCapsule.model_validate(record)
