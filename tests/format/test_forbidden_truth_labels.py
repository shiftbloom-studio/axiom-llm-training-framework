from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from hcaps.format.validation import validate_raw_axc_record


@pytest.mark.parametrize(
    "field_name",
    [
        "truth",
        "is_true",
        "is_correct",
        "correct",
        "label_correct",
        "factuality_label",
    ],
)
def test_forbidden_truth_style_labels_are_rejected(
    minimal_axc_record: Any,
    field_name: str,
) -> None:
    record = deepcopy(minimal_axc_record)
    record["training"][field_name] = True

    report = validate_raw_axc_record(record)

    assert not report.ok
    assert report.issues[0].code == "forbidden_truth_field"
    assert field_name in report.issues[0].message
