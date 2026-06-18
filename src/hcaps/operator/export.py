"""Operator export helpers for reproducibility bundles."""

from __future__ import annotations

from pathlib import Path

from hcaps.experiments.artifacts import write_reproducibility_bundle


def export_run(run_dir: str | Path, *, output: str | Path | None = None) -> Path:
    return write_reproducibility_bundle(Path(run_dir), output=Path(output) if output else None)
