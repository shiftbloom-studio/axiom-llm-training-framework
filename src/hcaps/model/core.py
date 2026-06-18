"""Full-complexity structured-native core blocks."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from hcaps.model.config import AxiomModelConfig
from hcaps.model.context import LateralContextEncoder
from hcaps.model.geometry_hooks import GeometryContext
from hcaps.model.provider import ProviderContextEncoder
from hcaps.model.temporal import TemporalContextEncoder
from hcaps.model.types import AxiomModelInput, CoreOutput, EncoderOutput


class ClaimFieldBlock(nn.Module):
    """Typed-slot attention block with context conditioning."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.model_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(config.model_dim)
        self.context_gate = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 4),
            _activation(config),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim * 4, config.model_dim),
        )
        self.norm_mlp = nn.LayerNorm(config.model_dim)

    def forward(self, slot_states: Tensor, slot_mask: Tensor, conditioning: Tensor) -> Tensor:
        attended, _ = self.attention(
            slot_states,
            slot_states,
            slot_states,
            key_padding_mask=~slot_mask,
            need_weights=False,
        )
        hidden = self.norm_attention(slot_states + attended)
        gated_context = self.context_gate(conditioning).unsqueeze(1) * conditioning.unsqueeze(1)
        hidden = hidden + gated_context
        hidden = self.norm_mlp(hidden + self.mlp(hidden))
        return torch.where(slot_mask.unsqueeze(-1), hidden, slot_states)


class FullComplexityCore(nn.Module):
    """Structured reasoning core with P5 geometry hook points."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.config = config
        self.temporal_encoder = TemporalContextEncoder(config)
        self.lateral_encoder = LateralContextEncoder(config)
        self.provider_encoder = ProviderContextEncoder(config)
        layer_count = config.num_layers if config.core_type == "claim_field_transformer" else 1
        self.blocks = nn.ModuleList(ClaimFieldBlock(config) for _ in range(layer_count))
        self.context_merge = nn.Sequential(
            nn.Linear(config.model_dim * 6, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.GELU(),
        )

    def forward(
        self,
        encoder_output: EncoderOutput,
        model_input: AxiomModelInput,
        *,
        relation_state: Tensor,
        provenance_state: Tensor,
        geometry_context: GeometryContext,
    ) -> CoreOutput:
        temporal_state = self.temporal_encoder(model_input)
        lateral_state = self.lateral_encoder(model_input)
        provider_state = self.provider_encoder(model_input)
        geometry_state = (
            geometry_context.conditioning
            if geometry_context.conditioning is not None
            else torch.zeros_like(temporal_state)
        )
        conditioning = self.context_merge(
            torch.cat(
                [
                    relation_state,
                    provenance_state,
                    temporal_state,
                    lateral_state,
                    provider_state,
                    geometry_state,
                ],
                dim=-1,
            )
        )
        slot_states = encoder_output.slot_states
        for block in self.blocks:
            slot_states = block(slot_states, encoder_output.slot_mask, conditioning)

        core_encoder = replace(encoder_output, slot_states=slot_states)
        claim_state_vector = _pool_or_zero(core_encoder, "claim_state")
        context_state = self.context_merge(
            torch.cat(
                [
                    relation_state,
                    provenance_state,
                    temporal_state,
                    lateral_state,
                    provider_state,
                    geometry_state,
                ],
                dim=-1,
            )
        )
        return CoreOutput(
            slot_states=slot_states,
            slot_type_ids=encoder_output.slot_type_ids,
            slot_mask=encoder_output.slot_mask,
            source_groups=encoder_output.source_groups,
            claim_state_vector=claim_state_vector,
            relation_state=relation_state,
            provenance_state=provenance_state,
            context_state=context_state,
            router_state=claim_state_vector + context_state,
            diagnostics={
                "core_type": self.config.core_type,
                "layer_count": len(self.blocks),
                "core_slot_shape": tuple(slot_states.shape),
                "geometry_hook_used": geometry_context.conditioning is not None,
                "context_state_shape": tuple(context_state.shape),
            },
        )


def _activation(config: AxiomModelConfig) -> nn.Module:
    if config.activation == "relu":
        return nn.ReLU()
    if config.activation == "silu":
        return nn.SiLU()
    return nn.GELU()


def _pool_or_zero(encoder_output: EncoderOutput, group_name: str) -> Tensor:
    indices = [
        index for index, name in enumerate(encoder_output.source_groups) if name == group_name
    ]
    if not indices:
        return torch.zeros(
            (encoder_output.slot_states.shape[0], encoder_output.slot_states.shape[-1]),
            dtype=encoder_output.slot_states.dtype,
            device=encoder_output.slot_states.device,
        )
    index_tensor = torch.tensor(indices, dtype=torch.long, device=encoder_output.slot_states.device)
    values = encoder_output.slot_states.index_select(1, index_tensor)
    masks = encoder_output.slot_mask.index_select(1, index_tensor).unsqueeze(-1).to(values.dtype)
    denominator = masks.sum(dim=1).clamp_min(1.0)
    return (values * masks).sum(dim=1) / denominator
