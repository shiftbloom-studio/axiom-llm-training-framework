"""Reusable validation helpers for Axiom claim-state schemas."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime


def ensure_timezone_aware(timestamp: datetime, *, field_name: str) -> datetime:
    """Require a timezone-aware datetime and return it unchanged."""

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        msg = f"{field_name} must be timezone-aware"
        raise ValueError(msg)
    return timestamp


def require_not_after_cutoff(
    timestamps: Iterable[datetime],
    *,
    cutoff: datetime,
    label: str,
) -> None:
    """Reject input timestamps after the temporal cutoff."""

    for timestamp in timestamps:
        if timestamp > cutoff:
            msg = (
                f"temporal leakage: {label} timestamp {timestamp.isoformat()} "
                f"is after cutoff {cutoff.isoformat()}"
            )
            raise ValueError(msg)


def require_future_targets_after_cutoff(
    timestamps: Iterable[datetime],
    *,
    cutoff: datetime,
    label: str,
) -> None:
    """Reject future-facing target timestamps at or before the temporal cutoff."""

    for timestamp in timestamps:
        if timestamp <= cutoff:
            msg = (
                f"target leakage: {label} timestamp {timestamp.isoformat()} "
                f"is not after cutoff {cutoff.isoformat()}"
            )
            raise ValueError(msg)
