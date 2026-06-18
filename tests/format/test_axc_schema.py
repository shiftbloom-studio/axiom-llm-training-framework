from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from hcaps.format.capsule import AxcCapsule
from hcaps.format.validation import validate_raw_axc_record


def test_minimal_valid_axc_capsule_validates(minimal_axc_record: Any) -> None:
    capsule = AxcCapsule.model_validate(minimal_axc_record)

    assert capsule.format == "AXC"
    assert capsule.claim.canonical_text == "Aspirin inhibits cyclooxygenase enzymes."


def test_empty_canonical_claim_text_fails(minimal_axc_record: Any) -> None:
    record = deepcopy(minimal_axc_record)
    record["claim"]["canonical_text"] = ""

    with pytest.raises(ValidationError, match=r"claim\.canonical_text"):
        AxcCapsule.model_validate(record)


def test_forbidden_truth_fields_fail_before_model_coercion(minimal_axc_record: Any) -> None:
    record = deepcopy(minimal_axc_record)
    record["truth"] = True

    report = validate_raw_axc_record(record)

    assert not report.ok
    assert report.issues[0].code == "forbidden_truth_field"


def test_relation_confidence_outside_unit_interval_fails(minimal_axc_record: Any) -> None:
    record = deepcopy(minimal_axc_record)
    record["relations"][0]["confidence"] = 1.2

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        AxcCapsule.model_validate(record)


def test_probability_epistemic_value_outside_unit_interval_fails(
    minimal_axc_record: Any,
) -> None:
    record = deepcopy(minimal_axc_record)
    record["epistemic_state"]["uncertainty"]["value"] = -0.1

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AxcCapsule.model_validate(record)


def test_independent_redundancy_allows_above_one_and_rejects_negative(
    minimal_axc_record: Any,
) -> None:
    record = deepcopy(minimal_axc_record)
    record["epistemic_state"]["independent_redundancy"]["effective_count"] = 4.2
    assert AxcCapsule.model_validate(record)

    record["epistemic_state"]["independent_redundancy"]["effective_count"] = -1.0
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AxcCapsule.model_validate(record)


def test_raw_geometry_matrices_are_rejected(minimal_axc_record: Any) -> None:
    record = deepcopy(minimal_axc_record)
    record["geometry"]["raw_connection_matrix"] = [[1.0, 0.0], [0.0, 1.0]]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AxcCapsule.model_validate(record)
