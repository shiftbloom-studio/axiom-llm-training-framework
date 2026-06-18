from __future__ import annotations

from pathlib import Path

import pytest

from hcaps.format.streams import validate_axc_stream


@pytest.mark.parametrize(
    "filename,expected_code,expected_message",
    [
        (
            "future_leakage_capsule.axc",
            "axc_schema_validation_failed",
            "temporal leakage",
        ),
        (
            "missing_provenance_capsule.axc",
            "axc_schema_validation_failed",
            "provenance reference missing",
        ),
        (
            "forbidden_truth_label_capsule.axc",
            "forbidden_truth_field",
            "truth",
        ),
        (
            "unsupported_version_capsule.axc",
            "unsupported_format_version",
            "unsupported AXC format_version",
        ),
    ],
)
def test_invalid_axc_examples_fail_for_expected_reason(
    axf_examples_dir: Path,
    filename: str,
    expected_code: str,
    expected_message: str,
) -> None:
    report = validate_axc_stream(axf_examples_dir / "invalid" / filename)

    assert not report.ok
    assert report.issues
    assert report.issues[0].code == expected_code
    assert expected_message in report.issues[0].message
    assert report.issues[0].line_number == 1
