"""Provider-context conditioning for data-ingress provenance signals."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from hcaps.model.config import AxiomModelConfig
from hcaps.model.features import GroupFeatureProjector
from hcaps.model.types import AxiomModelInput


class ProviderContextEncoder(nn.Module):
    """Encode provider identity/slant and cache/replay context when enabled."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.config = config
        self.projector = GroupFeatureProjector(config)

    def forward(self, model_input: AxiomModelInput) -> Tensor:
        disabled = (
            not self.config.use_provider_context
            or self.config.ablations.no_provider_context
            or self.config.ablations.no_side_channels
        )
        if disabled:
            return torch.zeros(
                (model_input.batch_size, self.projector.output_dim),
                dtype=torch.float32,
                device=model_input.device,
            )
        return cast(
            Tensor,
            self.projector(
                model_input.group("provider_context"),
                batch_size=model_input.batch_size,
                device=model_input.device,
            ),
        )
