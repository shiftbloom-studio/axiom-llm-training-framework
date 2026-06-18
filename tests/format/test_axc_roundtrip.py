from __future__ import annotations

from pathlib import Path

import orjson

from hcaps.format.capsule import AxcCapsule
from hcaps.format.hashing import canonical_json_bytes
from hcaps.format.streams import read_axc_stream, write_axc_stream


def test_axc_capsule_canonical_json_roundtrip(axf_examples_dir: Path) -> None:
    capsule = next(read_axc_stream(axf_examples_dir / "minimal_capsules.axc"))
    canonical = canonical_json_bytes(capsule.model_dump(mode="json"))

    parsed = AxcCapsule.model_validate_json(canonical)
    recanonical = canonical_json_bytes(parsed.model_dump(mode="json"))

    assert parsed == capsule
    assert recanonical == canonical
    assert canonical.endswith(b"}")


def test_axc_stream_roundtrip_preserves_order_ids_and_records(
    axf_examples_dir: Path,
    tmp_path: Path,
) -> None:
    capsules = list(read_axc_stream(axf_examples_dir / "full_capsules.axc"))
    output = tmp_path / "roundtrip.axc"

    count = write_axc_stream(output, capsules)
    reloaded = list(read_axc_stream(output))

    assert count == len(capsules)
    assert [capsule.ids.capsule_id for capsule in reloaded] == [
        capsule.ids.capsule_id for capsule in capsules
    ]
    assert [orjson.loads(line) for line in output.read_bytes().splitlines() if line.strip()] == [
        capsule.model_dump(mode="json") for capsule in capsules
    ]
