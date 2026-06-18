"""Optimizer and scheduler construction for P6."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from hcaps.training.config import TrainingConfig
from hcaps.training.schedules import linear_warmup_decay_lambda


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Create the configured optimizer over trainable parameters."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer: Optimizer, config: TrainingConfig) -> LRScheduler | None:
    """Create an optional bounded scheduler."""

    if config.scheduler == "none":
        return None

    def lr_lambda(step: int) -> float:
        return linear_warmup_decay_lambda(
            step=step,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            cooldown_steps=config.cooldown_steps,
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
