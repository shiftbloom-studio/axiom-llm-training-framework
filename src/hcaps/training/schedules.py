"""Learning-rate schedule helpers."""

from __future__ import annotations


def linear_warmup_decay_lambda(
    *,
    step: int,
    max_steps: int,
    warmup_steps: int,
    cooldown_steps: int,
) -> float:
    """Compute a bounded warmup/decay multiplier."""

    if warmup_steps > 0 and step < warmup_steps:
        return max(0.0, float(step + 1) / float(warmup_steps))
    decay_start = max(warmup_steps, max_steps - cooldown_steps)
    if cooldown_steps > 0 and step >= decay_start:
        remaining = max(0, max_steps - step)
        return max(0.0, float(remaining) / float(cooldown_steps))
    return 1.0
