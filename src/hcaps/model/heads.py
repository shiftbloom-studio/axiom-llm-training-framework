"""Prediction heads for P4 structured-native outputs."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from hcaps.model.config import AxiomModelConfig


class ClaimStateHead(nn.Module):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.model_dim, config.axc_out_vocab_sizes["claim_state"])

    def forward(self, latent_state: Tensor) -> Tensor:
        return cast(Tensor, self.linear(latent_state))


class RelationPredictionHead(nn.Module):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.config = config
        self.type_head = nn.Linear(config.model_dim, config.axc_out_vocab_sizes["relation_type"])
        self.target_head = nn.Linear(
            config.model_dim,
            config.axc_out_vocab_sizes["relation_target"],
        )

    def forward(self, latent_state: Tensor) -> tuple[Tensor, Tensor]:
        expanded = latent_state.unsqueeze(1).expand(-1, self.config.max_relation_neighbors, -1)
        return cast(Tensor, self.type_head(expanded)), cast(Tensor, self.target_head(expanded))


class ProvenanceRecoveryHead(nn.Module):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.config = config
        self.source_head = nn.Linear(
            config.model_dim,
            config.axc_out_vocab_sizes["provenance_source"],
        )
        self.span_head = nn.Linear(config.model_dim, config.axc_out_vocab_sizes["evidence_span"])

    def forward(self, latent_state: Tensor) -> tuple[Tensor, Tensor]:
        expanded = latent_state.unsqueeze(1).expand(-1, self.config.max_provenance_sources, -1)
        return cast(Tensor, self.source_head(expanded)), cast(Tensor, self.span_head(expanded))


class EpistemicRegressionHead(nn.Module):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.status_head = nn.Linear(
            config.model_dim,
            config.axc_out_vocab_sizes["epistemic_status"],
        )
        self.value_head = nn.Sequential(nn.Linear(config.model_dim, 4), nn.Tanh())

    def forward(self, latent_state: Tensor) -> tuple[Tensor, Tensor]:
        return cast(Tensor, self.status_head(latent_state)), cast(
            Tensor,
            self.value_head(latent_state),
        )


class StabilityPredictionHead(nn.Module):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.model_dim, config.axc_out_vocab_sizes["stability"])

    def forward(self, latent_state: Tensor) -> Tensor:
        return cast(Tensor, self.linear(latent_state))


class FutureSummaryLatentHead(nn.Module):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.model_dim, config.axc_out_vocab_sizes["future_summary"])

    def forward(self, latent_state: Tensor) -> Tensor:
        return cast(Tensor, self.linear(latent_state))


class GeometryObservableHead(nn.Module):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.model_dim, config.geometry_observable_dim)

    def forward(self, latent_state: Tensor) -> Tensor:
        return cast(Tensor, self.linear(latent_state))


class TextProjectionHead(nn.Module):
    """Secondary token projection head for LLM comparability."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.text_vocab_size, config.model_dim)
        self.norm = nn.LayerNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, config.text_vocab_size)

    def forward(
        self,
        latent_state: Tensor,
        *,
        text_input_ids: Tensor | None,
        text_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if text_input_ids is None:
            length = self.config.max_text_length
            hidden = latent_state.unsqueeze(1).expand(-1, length, -1)
        else:
            ids = torch.remainder(
                torch.abs(text_input_ids.to(dtype=torch.long)),
                self.config.text_vocab_size,
            )
            hidden = self.token_embedding(ids)
            hidden = hidden + latent_state.unsqueeze(1)
        if text_attention_mask is not None:
            hidden = hidden * text_attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return cast(Tensor, self.output(self.norm(hidden)))
