from __future__ import annotations

from hcaps.format.validation import forbidden_truth_field_issues


def test_nested_forbidden_truth_fields_are_detected() -> None:
    issues = forbidden_truth_field_issues({"claim": {"is_correct": True}})

    assert issues
    assert issues[0].path == "claim.is_correct"
