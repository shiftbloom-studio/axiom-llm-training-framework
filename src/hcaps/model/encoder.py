"""Structured encoder over typed claim-field slots."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hcaps.model.config import AxiomModelConfig
from hcaps.model.types import SLOT_TYPE_IDS, EncoderOutput, TypedSlotBundle


class StructuredEncoder(nn.Module):
    """Encode typed structured slots without serializing them to plain text."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.config = config
        self.type_embedding = nn.Embedding(len(SLOT_TYPE_IDS), config.slot_dim)
        self.input_projection = (
            nn.Identity()
            if config.slot_dim == config.model_dim
            else nn.Linear(config.slot_dim, config.model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.model_dim * 4,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, config.num_layers // 2))
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, slots: TypedSlotBundle) -> EncoderOutput:
        hidden = slots.slot_tensor + self.type_embedding(slots.slot_type_ids)
        hidden = self.input_projection(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=~slots.slot_mask)
        hidden = self.norm(hidden)
        zero = torch.zeros(
            (hidden.shape[0], hidden.shape[-1]),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        return EncoderOutput(
            slot_states=hidden,
            slot_type_ids=slots.slot_type_ids,
            slot_mask=slots.slot_mask,
            source_groups=slots.source_groups,
            claim_state_vector=_group_pool(hidden, slots, "claim_state", fallback=zero),
            epistemic_vector=_group_pool(hidden, slots, "epistemic_state", fallback=zero),
            relation_context_vector=_group_pool(hidden, slots, "relations", fallback=zero),
            provenance_context_vector=_group_pool(hidden, slots, "provenance", fallback=zero),
            temporal_vector=_group_pool(hidden, slots, "temporal_scope", fallback=zero),
            lateral_context_vector=_group_pool(hidden, slots, "lateral_context", fallback=zero),
            provider_context_vector=_group_pool(hidden, slots, "provider_context", fallback=zero),
            geometry_context_vector=_group_pool(hidden, slots, "geometry_features", fallback=zero),
            diagnostics={
                "encoder_slot_shape": tuple(hidden.shape),
                "slot_types_preserved": True,
                "source_groups": slots.source_groups,
            },
        )


def _group_pool(
    slot_states: Tensor,
    slots: TypedSlotBundle,
    group_name: str,
    *,
    fallback: Tensor,
) -> Tensor:
    slot_indices = [index for index, name in enumerate(slots.source_groups) if name == group_name]
    if not slot_indices:
        return fallback
    index_tensor = torch.tensor(slot_indices, dtype=torch.long, device=slot_states.device)
    values = slot_states.index_select(1, index_tensor)
    masks = slots.slot_mask.index_select(1, index_tensor).unsqueeze(-1).to(values.dtype)
    denominator = masks.sum(dim=1).clamp_min(1.0)
    return (values * masks).sum(dim=1) / denominator
