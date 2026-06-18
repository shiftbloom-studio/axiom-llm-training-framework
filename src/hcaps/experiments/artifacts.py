"""Run artifact helpers for P6 experiment suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hcaps.format.hashing import file_hash
from hcaps.training.logging import write_json


def write_reproducibility_bundle(run_dir: Path, *, output: Path | None = None) -> Path:
    """Write a hash manifest over core P6 run artifacts."""

    target = output or run_dir / "exports" / "reproducibility_bundle.json"
    artifacts: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and ".pt" not in path.suffixes:
            artifacts[str(path.relative_to(run_dir))] = file_hash(path)
    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "artifact_hashes": artifacts,
        "provider_secrets_included": False,
        "external_llm_calls_in_training_or_evaluation": False,
    }
    write_json(target, payload)
    return target
