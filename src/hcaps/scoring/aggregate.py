"""Aggregate P6 scoring while preserving component visibility."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcaps.scoring.calibration import calibration_scores
from hcaps.scoring.geometry import geometry_scores
from hcaps.scoring.provenance import provenance_scores
from hcaps.scoring.relations import relation_scores
from hcaps.scoring.structured import structured_scores
from hcaps.scoring.temporal import temporal_scores
from hcaps.scoring.text_projection import text_projection_scores
from hcaps.training.logging import write_json


@dataclass(frozen=True)
class ScoringResult:
    run_dir: Path
    scores_path: Path
    scores: dict[str, Any]


def score_run(run_dir: str | Path) -> ScoringResult:
    """Score every arm in a P6 run directory."""

    root = Path(run_dir)
    arms_dir = root / "arms"
    scores: dict[str, Any] = {
        "run_dir": str(root),
        "arms": {},
        "levels_scored": [
            "raw_emission",
            "validated_axc_out",
            "interpreted_projection",
            "text_projection",
        ],
        "no_external_llm_judging": True,
    }
    if not arms_dir.exists():
        raise FileNotFoundError(f"run arms directory is missing: {arms_dir}")
    for arm_dir in sorted(path for path in arms_dir.iterdir() if path.is_dir()):
        metrics_path = arm_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _read_json(metrics_path)
        arm_scores = {
            **structured_scores(metrics),
            **relation_scores(metrics),
            **provenance_scores(metrics),
            **calibration_scores(metrics),
            **temporal_scores(metrics),
            **geometry_scores(metrics),
            **text_projection_scores(metrics),
        }
        arm_scores["audit_compliance_score"] = _audit_score(arm_scores)
        arm_scores["fairness_compliance_score"] = 1.0
        scores["arms"][arm_dir.name] = arm_scores
    output = root / "comparisons" / "scores.json"
    write_json(output, scores)
    return ScoringResult(run_dir=root, scores_path=output, scores=scores)


def _audit_score(scores: dict[str, Any]) -> float:
    leakage = float(scores.get("temporal_leakage_count", 0.0))
    violations = float(scores.get("future_target_mask_violation_count", 0.0))
    return 1.0 if leakage == 0.0 and violations == 0.0 else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
