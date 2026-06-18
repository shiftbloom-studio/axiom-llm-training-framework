"""Corpus construction quality tiers."""

from __future__ import annotations

from typing import Literal

QualityTier = Literal["bronze", "silver", "gold", "quarantine"]
QUARANTINE_CONFIDENCE_THRESHOLD = 0.2
SILVER_CONFIDENCE_THRESHOLD = 0.72


def quality_tier(
    *,
    confidence: float,
    warning_count: int,
    provider_disagreement_count: int,
    human_reviewed: bool = False,
) -> QualityTier:
    """Classify construction quality, not claim truth."""

    if human_reviewed:
        return "gold"
    if warning_count > 1 or confidence < QUARANTINE_CONFIDENCE_THRESHOLD:
        return "quarantine"
    if confidence >= SILVER_CONFIDENCE_THRESHOLD and provider_disagreement_count == 0:
        return "silver"
    return "bronze"
