"""AXC stream read/write utilities."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import orjson
from pydantic import ValidationError

from hcaps.format.capsule import AxcCapsule
from hcaps.format.validation import ValidationIssue, ValidationReport, validate_raw_axc_record


def write_axc_stream(path: str | Path, capsules: Iterable[AxcCapsule]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("wb") as handle:
        for capsule in capsules:
            validated = AxcCapsule.model_validate(capsule)
            handle.write(
                orjson.dumps(validated.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
            )
            handle.write(b"\n")
            count += 1
    return count


def read_axc_stream(path: str | Path) -> Iterator[AxcCapsule]:
    input_path = Path(path)
    with input_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield AxcCapsule.model_validate_json(line)
            except (ValidationError, ValueError) as exc:
                msg = f"{input_path}:{line_number}: invalid AXC record: {exc}"
                raise ValueError(msg) from exc


def validate_axc_stream(path: str | Path, *, strict: bool = True) -> ValidationReport:
    input_path = Path(path)
    valid_count = 0
    issues: list[ValidationIssue] = []
    with input_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = orjson.loads(line)
                report = validate_raw_axc_record(payload)
            except Exception as exc:
                report = ValidationReport(
                    ok=False,
                    issues=[
                        ValidationIssue(
                            code="invalid_json",
                            message=str(exc),
                            line_number=line_number,
                        )
                    ],
                )
            if report.ok:
                valid_count += 1
            else:
                for issue in report.issues:
                    issues.append(issue.model_copy(update={"line_number": line_number}))
                if strict:
                    continue
    return ValidationReport(ok=not issues, valid_count=valid_count, issues=issues)
