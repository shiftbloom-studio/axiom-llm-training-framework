"""Provenance-aware conditioning without hidden oracle behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from hcaps.model.config import AxiomModelConfig
from hcaps.model.features import GroupFeatureProjector
from hcaps.model.types import AxiomModelInput


@dataclass(frozen=True)
class ProvenanceConditioningOutput:
    slot_states: Tensor
    provenance_state: Tensor
    diagnostics: dict[str, Any]


class ProvenanceConditioner(nn.Module):
    """Condition claim-state slots on source/evidence tensors, when enabled."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.config = config
        self.source_projector = GroupFeatureProjector(config)
        self.temporal_projector = GroupFeatureProjector(config)
        self.gate = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(
        self,
        slot_states: Tensor,
        slot_mask: Tensor,
        model_input: AxiomModelInput,
    ) -> ProvenanceConditioningOutput:
        disabled = (
            not self.config.use_provenance_conditioning
            or self.config.ablations.no_provenance
            or self.config.ablations.no_side_channels
            or self.config.ablations.text_only
        )
        zero = torch.zeros(
            (model_input.batch_size, self.config.model_dim),
            dtype=slot_states.dtype,
            device=slot_states.device,
        )
        if disabled:
            return ProvenanceConditioningOutput(
                slot_states=slot_states,
                provenance_state=zero,
                diagnostics={"enabled": False, "hidden_oracle_repairs": False},
            )

        provenance = self.source_projector(
            model_input.group("provenance"),
            batch_size=model_input.batch_size,
            device=model_input.device,
        )
        temporal = self.temporal_projector(
            model_input.group("temporal"),
            batch_size=model_input.batch_size,
            device=model_input.device,
        )
        gate = self.gate(torch.cat([provenance, temporal], dim=-1)).unsqueeze(1)
        conditioned = self.norm(slot_states + gate * provenance.unsqueeze(1))
        conditioned = torch.where(slot_mask.unsqueeze(-1), conditioned, slot_states)
        return ProvenanceConditioningOutput(
            slot_states=conditioned,
            provenance_state=provenance,
            diagnostics={
                "enabled": True,
                "provenance_state_shape": tuple(provenance.shape),
                "hidden_oracle_repairs": False,
            },
        )
