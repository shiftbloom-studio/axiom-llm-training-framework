from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle
from hcaps.axt.validation import validate_axt_bundle


def test_manifest_hashes_are_valid(compiled_minimal_axp: Path) -> None:
    bundle = AxtBundle(compiled_minimal_axp)

    assert bundle.validate_hashes()["ok"] is True
    assert validate_axt_bundle(compiled_minimal_axp)["ok"] is True
    assert "tensors/ids.safetensors" in bundle.manifest.hashes
