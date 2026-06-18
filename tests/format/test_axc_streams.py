from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import orjson

from hcaps.format.capsule import AxcCapsule
from hcaps.format.streams import read_axc_stream, validate_axc_stream, write_axc_stream


def test_axc_stream_roundtrip(minimal_axc_record: Any, tmp_path: Path) -> None:
    capsule = AxcCapsule.model_validate(minimal_axc_record)
    path = tmp_path / "capsules.axc"

    count = write_axc_stream(path, [capsule])
    roundtripped = list(read_axc_stream(path))

    assert count == 1
    assert roundtripped[0].ids.capsule_id == capsule.ids.capsule_id


def test_axc_stream_validation_reports_line_numbers(
    minimal_axc_record: Any,
    tmp_path: Path,
) -> None:
    invalid = deepcopy(minimal_axc_record)
    invalid["claim"]["canonical_text"] = ""
    path = tmp_path / "capsules.axc"
    path.write_bytes(orjson.dumps(minimal_axc_record) + b"\n" + orjson.dumps(invalid) + b"\n")

    report = validate_axc_stream(path)

    assert not report.ok
    assert report.valid_count == 1
    assert report.issues[0].line_number == 2
