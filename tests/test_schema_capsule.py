from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import orjson
import pytest
from pydantic import ValidationError

from hcaps.schema.capsule import HoloCapsule
from hcaps.store.jsonl import JsonlCapsuleStore

FIXTURES = Path(__file__).parent / "fixtures"


def _valid_record() -> dict[str, Any]:
    line = (FIXTURES / "valid_capsules.jsonl").read_bytes().splitlines()[0]
    record = orjson.loads(line)
    assert isinstance(record, dict)
    return record


def test_valid_fixtures_load_successfully() -> None:
    capsules = list(JsonlCapsuleStore().read_capsules(FIXTURES / "valid_capsules.jsonl"))

    assert [capsule.capsule_id for capsule in capsules] == [
        "cap_ulcer_hpylori_001",
        "cap_aspirin_cox_001",
    ]


def test_relation_enum_rejects_unknown_relation_types() -> None:
    record = _valid_record()
    record["relations"][0]["relation_type"] = "merely_mentions"

    with pytest.raises(ValidationError, match="relation_type"):
        HoloCapsule.model_validate(record)


def test_epistemic_values_outside_unit_interval_fail() -> None:
    record = _valid_record()
    record["epistemic_state"]["uncertainty"] = 1.2

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        HoloCapsule.model_validate(record)


def test_capsule_without_provenance_fails() -> None:
    record = _valid_record()
    record["provenance"] = []

    with pytest.raises(ValidationError, match="provenance"):
        HoloCapsule.model_validate(record)


def test_surface_forms_are_required() -> None:
    record = deepcopy(_valid_record())
    record["surface_forms"] = {
        "primary_text": None,
        "alternate_texts": [],
        "original_spans": [],
        "summary": None,
        "teaching_note": None,
        "counterargument": None,
        "table_form": None,
    }

    with pytest.raises(ValidationError, match="at least one surface form"):
        HoloCapsule.model_validate(record)


def test_redundancy_basis_rejects_raw_popularity() -> None:
    record = _valid_record()
    record["epistemic_state"]["redundancy_basis"] = "raw popularity count"

    with pytest.raises(ValidationError, match="independent-community"):
        HoloCapsule.model_validate(record)
