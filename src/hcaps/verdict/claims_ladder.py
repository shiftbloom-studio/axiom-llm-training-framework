"""Axiom P6 claims ladder without HKR proof overclaiming."""

from __future__ import annotations

from typing import Any

CLAIMS_LADDER: tuple[dict[str, Any], ...] = (
    {"level": 0, "claim": "Infrastructure works."},
    {"level": 1, "claim": "AXC/AXT/AXC-out training is technically feasible."},
    {"level": 2, "claim": "Structured arms beat flat text on at least one epistemic metric."},
    {
        "level": 3,
        "claim": "Structured-native beats structured text under matched source/compute/params.",
    },
    {
        "level": 4,
        "claim": "Geometry adds predictive value beyond non-geometric structured baseline.",
    },
    {
        "level": 5,
        "claim": "Geometry effects survive context/provider shuffle and popularity controls.",
    },
    {
        "level": 6,
        "claim": "Results generalize across splits/domains and support HKR-inspired dynamics.",
    },
)
MAX_CLAIMS_LEVEL = 6


def claims_ladder_payload(reached_level: int) -> dict[str, Any]:
    if reached_level > MAX_CLAIMS_LEVEL:
        raise ValueError("claims ladder cannot exceed Level 6; HKR proof claims are forbidden")
    return {
        "reached_level": reached_level,
        "levels": list(CLAIMS_LADDER),
        "forbidden_level_7": "HKR is proven.",
        "proof_language_emitted": False,
    }
