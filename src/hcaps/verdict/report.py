"""Verdict report rendering for P6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hcaps.training.logging import write_json
from hcaps.verdict.evidence_tables import branch_verdict_table
from hcaps.verdict.research_card import write_research_card


def write_verdict_report(
    run_dir: str | Path,
    *,
    verdict: dict[str, Any],
    output: str | Path | None = None,
) -> Path:
    root = Path(run_dir)
    verdict_dir = root / "verdict"
    verdict_dir.mkdir(parents=True, exist_ok=True)
    verdict_json = verdict_dir / "verdict.json"
    claims_json = verdict_dir / "claims_ladder.json"
    write_json(verdict_json, verdict)
    claims = verdict.get("claims_ladder", {})
    write_json(claims_json, claims if isinstance(claims, dict) else {})
    report_path = Path(output) if output is not None else verdict_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown_report(verdict), encoding="utf-8")
    write_research_card(verdict_dir / "research_card.md", verdict)
    return report_path


def inspect_verdict(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"verdict file must contain a JSON object: {path}")
    return payload


def _markdown_report(verdict: dict[str, Any]) -> str:
    ladder = verdict.get("claims_ladder", {})
    reached = ladder.get("reached_level", 0) if isinstance(ladder, dict) else 0
    return "\n".join(
        [
            "# Axiom P6 Verdict Report",
            "",
            f"Overall verdict: `{verdict.get('overall_verdict', 'inconclusive')}`",
            "",
            "This report is a bounded research-runtime verdict. It does not claim benchmark",
            "performance and does not claim HKR proof.",
            "",
            branch_verdict_table(verdict),
            "",
            f"Claims ladder reached: Level {reached}",
            "",
            "External LLM judging used: `false`",
            "Text projection role: secondary projection and comparison interface.",
            "Structured output role: primary AXC-out emission surface.",
            "",
        ]
    )
