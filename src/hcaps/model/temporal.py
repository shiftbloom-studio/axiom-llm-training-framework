"""Temporal feature conditioning kept separate from lateral context."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from hcaps.model.config import AxiomModelConfig
from hcaps.model.features import GroupFeatureProjector
from hcaps.model.types import AxiomModelInput


class TemporalContextEncoder(nn.Module):
    """Encode time/cutoff features without mixing them into lateral context."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.projector = GroupFeatureProjector(config)

    def forward(self, model_input: AxiomModelInput) -> Tensor:
        if model_input.group("temporal"):
            return cast(
                Tensor,
                self.projector(
                    model_input.group("temporal"),
                    batch_size=model_input.batch_size,
                    device=model_input.device,
                ),
            )
        return torch.zeros(
            (model_input.batch_size, self.projector.output_dim),
            dtype=torch.float32,
            device=model_input.device,
        )
