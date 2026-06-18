"""Precision-mode helpers for CPU-safe smoke training."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch

from hcaps.training.config import PrecisionMode


def autocast_context(
    device: torch.device, precision: PrecisionMode
) -> AbstractContextManager[None]:
    """Return an autocast context for supported reduced precision modes."""

    if precision == "bf16" and device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if precision == "fp16" and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()
