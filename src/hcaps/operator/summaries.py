"""Human-readable summaries for P6 runs."""

from __future__ import annotations

from typing import Any


def run_summary_lines(summary: dict[str, Any]) -> list[str]:
    return [
        f"Run: {summary.get('run_id')}",
        f"Arms: {', '.join(str(arm) for arm in summary.get('arms', []))}",
        f"Scores: {summary.get('has_scores')}",
        f"Verdict: {summary.get('overall_verdict') or 'not generated'}",
    ]
