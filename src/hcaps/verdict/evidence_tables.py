"""Markdown evidence table helpers."""

from __future__ import annotations

from typing import Any


def branch_verdict_table(verdict: dict[str, Any]) -> str:
    rows = ["| Branch | Outcome | Reason |", "|---|---:|---|"]
    branches = verdict.get("branch_verdicts", {})
    if isinstance(branches, dict):
        for branch, payload in branches.items():
            if not isinstance(payload, dict):
                continue
            outcome = payload.get("outcome", "inconclusive")
            reason = payload.get("reason", "")
            rows.append(f"| {branch} | {outcome} | {reason} |")
    return "\n".join(rows)
