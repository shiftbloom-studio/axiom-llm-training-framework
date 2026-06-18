"""Parameter counting utilities for P4 model configs."""

from __future__ import annotations

from torch import nn


def parameter_count(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}


def parameter_count_text(model: nn.Module) -> str:
    counts = parameter_count(model)
    return f"total={counts['total']} trainable={counts['trainable']}"
