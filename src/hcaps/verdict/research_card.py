"""Research-card writer for P6 outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def research_card_text(verdict: dict[str, Any]) -> str:
    overall = verdict.get("overall_verdict", "inconclusive")
    return "\n".join(
        [
            "# Axiom P6 Research Card",
            "",
            f"Overall verdict: `{overall}`",
            "",
            "This card summarizes a bounded Axiom run. It is not a benchmark claim "
            "and not HKR proof.",
            "",
            "- Structured AXC-out outputs are scored separately from text projection.",
            "- Text projection is mandatory but secondary.",
            "- External LLM judging was not used.",
            "- Geometry evidence is treated as experimental and ablatable.",
            "",
        ]
    )


def write_research_card(path: str | Path, verdict: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(research_card_text(verdict), encoding="utf-8")
    return target
