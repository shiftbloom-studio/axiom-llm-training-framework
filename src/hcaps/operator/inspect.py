"""Run inspection helpers for the P6 operator surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def inspect_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    manifest = _read_optional(root / "run_manifest.json")
    verdict = _read_optional(root / "verdict" / "verdict.json")
    arms = (
        [path.name for path in sorted((root / "arms").iterdir()) if path.is_dir()]
        if (root / "arms").exists()
        else []
    )
    return {
        "run_dir": str(root),
        "run_id": manifest.get("run_id", root.name),
        "arms": arms,
        "has_scores": (root / "comparisons" / "scores.json").exists(),
        "has_verdict": bool(verdict),
        "overall_verdict": verdict.get("overall_verdict"),
        "provider_calls_during_training_or_evaluation": False,
    }


def list_runs(root: str | Path) -> list[dict[str, Any]]:
    run_root = Path(root)
    if not run_root.exists():
        return []
    return [inspect_run(path) for path in sorted(run_root.iterdir()) if path.is_dir()]


def _read_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
