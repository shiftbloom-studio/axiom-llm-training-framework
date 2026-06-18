"""AXF validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hcaps.format.capsule import AxcCapsule

FORBIDDEN_TRUTH_FIELDS = {
    "truth",
    "is_true",
    "true",
    "is_correct",
    "correct",
    "ground_truth",
    "label_truth",
    "label_correct",
    "factuality_label",
}


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str | None = None
    line_number: int | None = Field(default=None, ge=1)


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    valid_count: int = Field(default=0, ge=0)
    issues: list[ValidationIssue] = Field(default_factory=list)


def validate_raw_axc_record(record: dict[str, Any]) -> ValidationReport:
    issues = forbidden_truth_field_issues(record)
    if issues:
        return ValidationReport(ok=False, issues=issues)
    if record.get("format") != "AXC":
        return ValidationReport(
            ok=False,
            issues=[
                ValidationIssue(
                    code="unsupported_format",
                    message=f"unsupported AXC format {record.get('format')!r}",
                    path="format",
                )
            ],
        )
    if record.get("format_version") != "0.1.0":
        return ValidationReport(
            ok=False,
            issues=[
                ValidationIssue(
                    code="unsupported_format_version",
                    message=f"unsupported AXC format_version {record.get('format_version')!r}",
                    path="format_version",
                )
            ],
        )
    try:
        AxcCapsule.model_validate(record)
    except ValidationError as exc:
        return ValidationReport(
            ok=False,
            issues=[
                ValidationIssue(
                    code="axc_schema_validation_failed",
                    message=str(exc),
                )
            ],
        )
    return ValidationReport(ok=True, valid_count=1)


def validate_axc_capsule(capsule: AxcCapsule | dict[str, Any]) -> ValidationReport:
    record = capsule.model_dump(mode="json") if isinstance(capsule, AxcCapsule) else capsule
    return validate_raw_axc_record(record)


def validate_axc_stream(path: str | Path) -> ValidationReport:
    from hcaps.format.streams import validate_axc_stream as validate_stream  # noqa: PLC0415

    return validate_stream(path)


def validate_axp_package(path: str | Path) -> ValidationReport:
    from hcaps.format.package import validate_package  # noqa: PLC0415

    return validate_package(path)


def forbidden_truth_field_issues(record: Any, *, prefix: str = "") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if isinstance(record, dict):
        for key, value in record.items():
            path = f"{prefix}.{key}" if prefix else key
            if key.casefold() in FORBIDDEN_TRUTH_FIELDS:
                issues.append(
                    ValidationIssue(
                        code="forbidden_truth_field",
                        message=f"AXF records must not contain binary truth field {path!r}",
                        path=path,
                    )
                )
            issues.extend(forbidden_truth_field_issues(value, prefix=path))
    elif isinstance(record, list):
        for index, item in enumerate(record):
            path = f"{prefix}[{index}]"
            issues.extend(forbidden_truth_field_issues(item, prefix=path))
    return issues
