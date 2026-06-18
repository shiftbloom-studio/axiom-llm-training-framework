"""Relation and hypergraph neighborhood conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from hcaps.model.config import AxiomModelConfig
from hcaps.model.features import GroupFeatureProjector
from hcaps.model.types import AxiomModelInput


@dataclass(frozen=True)
class RelationConditioningOutput:
    slot_states: Tensor
    relation_state: Tensor
    diagnostics: dict[str, Any]


class RelationEdgeEncoder(nn.Module):
    """Encode relation, direction, confidence, and hard-negative features."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.relation_projector = GroupFeatureProjector(config)
        self.negative_projector = GroupFeatureProjector(config)
        self.merge = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.GELU(),
        )

    def forward(self, model_input: AxiomModelInput) -> Tensor:
        relation = self.relation_projector(
            model_input.group("relations"),
            batch_size=model_input.batch_size,
            device=model_input.device,
        )
        negatives = self.negative_projector(
            model_input.group("negative_samples"),
            batch_size=model_input.batch_size,
            device=model_input.device,
        )
        return cast(Tensor, self.merge(torch.cat([relation, negatives], dim=-1)))


class RelationNeighborhoodAggregator(nn.Module):
    """Aggregate binary-neighbor tensors while preserving an n-ary extension path."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.projector = GroupFeatureProjector(config)
        self.merge = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.GELU(),
        )

    def forward(self, relation_state: Tensor, model_input: AxiomModelInput) -> Tensor:
        neighborhood_state = self.projector(
            model_input.group("relation_neighborhoods"),
            batch_size=model_input.batch_size,
            device=model_input.device,
        )
        return cast(Tensor, self.merge(torch.cat([relation_state, neighborhood_state], dim=-1)))


class HyperedgePlaceholderAdapter(nn.Module):
    """Expose a clean extension point for n-ary/hyperedge tensors."""

    def forward(self, model_input: AxiomModelInput) -> dict[str, Any]:
        neighborhood = model_input.group("relation_neighborhoods")
        return {
            "hyperedge_path_available": any(key.startswith("hyperedge") for key in neighborhood),
            "supports_future_n_ary_tensors": True,
            "hardcoded_binary_only": False,
        }


class RelationConditionedAttention(nn.Module):
    """Condition typed slots on relation/neighborhood messages."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.model_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, slot_states: Tensor, slot_mask: Tensor, relation_state: Tensor) -> Tensor:
        memory = relation_state.unsqueeze(1)
        attended, _ = self.attention(slot_states, memory, memory, need_weights=False)
        gated = attended * self.gate(relation_state).unsqueeze(1)
        return cast(
            Tensor,
            self.norm(slot_states + gated * slot_mask.unsqueeze(-1).to(slot_states.dtype)),
        )


class RelationNeighborhoodConditioner(nn.Module):
    """Relation tensors + neighbor tensors -> relation-conditioned slot states."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.config = config
        self.edge_encoder = RelationEdgeEncoder(config)
        self.aggregator = RelationNeighborhoodAggregator(config)
        self.conditioned_attention = RelationConditionedAttention(config)
        self.hyperedge_adapter = HyperedgePlaceholderAdapter()

    def forward(
        self,
        slot_states: Tensor,
        slot_mask: Tensor,
        model_input: AxiomModelInput,
    ) -> RelationConditioningOutput:
        disabled = (
            not self.config.use_relation_conditioning
            or self.config.ablations.no_relations
            or self.config.ablations.text_only
        )
        if disabled:
            relation_state = torch.zeros(
                (model_input.batch_size, self.config.model_dim),
                dtype=slot_states.dtype,
                device=slot_states.device,
            )
            return RelationConditioningOutput(
                slot_states=slot_states,
                relation_state=relation_state,
                diagnostics={
                    "enabled": False,
                    "reason": "relation conditioning disabled by config/ablation",
                    "hypergraph": self.hyperedge_adapter(model_input),
                },
            )

        relation_state = self.edge_encoder(model_input)
        if not self.config.ablations.relation_neighborhood_off:
            relation_state = self.aggregator(relation_state, model_input)
        conditioned = self.conditioned_attention(slot_states, slot_mask, relation_state)
        return RelationConditioningOutput(
            slot_states=conditioned,
            relation_state=relation_state,
            diagnostics={
                "enabled": True,
                "relation_state_shape": tuple(relation_state.shape),
                "relation_neighborhood_off": self.config.ablations.relation_neighborhood_off,
                "hypergraph": self.hyperedge_adapter(model_input),
            },
        )
