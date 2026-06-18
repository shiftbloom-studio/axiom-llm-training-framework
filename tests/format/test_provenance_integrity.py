from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from hcaps.format.capsule import AxcCapsule


def test_source_spans_must_reference_provenance_sources(minimal_axc_record: Any) -> None:
    record = deepcopy(minimal_axc_record)
    record["surface_forms"]["source_spans"][0]["source_id"] = "src:fixture:" + ("a" * 24)

    with pytest.raises(ValidationError, match="provenance reference missing"):
        AxcCapsule.model_validate(record)


def test_relation_evidence_spans_must_reference_surface_spans(
    minimal_axc_record: Any,
) -> None:
    record = deepcopy(minimal_axc_record)
    record["relations"][0]["evidence_span_ids"] = ["span:" + ("a" * 24)]

    with pytest.raises(ValidationError, match="relation evidence_span_ids"):
        AxcCapsule.model_validate(record)
