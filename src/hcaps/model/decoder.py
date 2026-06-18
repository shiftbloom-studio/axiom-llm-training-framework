"""Structured decoder that emits native AXC-out raw tensors."""

from __future__ import annotations

from torch import Tensor, nn

from hcaps.model.axc_out import AXCOutRawEmission
from hcaps.model.config import AxiomModelConfig
from hcaps.model.heads import (
    ClaimStateHead,
    EpistemicRegressionHead,
    FutureSummaryLatentHead,
    GeometryObservableHead,
    ProvenanceRecoveryHead,
    RelationPredictionHead,
    StabilityPredictionHead,
)
from hcaps.model.types import AxiomModelInput


class StructuredDecoder(nn.Module):
    """Decode core/router latents into structured AXC-out raw emissions."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.config = config
        self.claim_state_head = ClaimStateHead(config)
        self.relation_head = RelationPredictionHead(config)
        self.provenance_head = ProvenanceRecoveryHead(config)
        self.epistemic_head = EpistemicRegressionHead(config)
        self.stability_head = StabilityPredictionHead(config)
        self.future_summary_head = FutureSummaryLatentHead(config)
        self.geometry_head = GeometryObservableHead(config)

    def forward(self, latent_state: Tensor, model_input: AxiomModelInput) -> AXCOutRawEmission:
        relation_type_logits, relation_target_logits = self.relation_head(latent_state)
        provenance_source_logits, evidence_span_logits = self.provenance_head(latent_state)
        epistemic_status_logits, epistemic_values = self.epistemic_head(latent_state)
        uncertainty_values = epistemic_values[:, :1]
        return AXCOutRawEmission(
            claim_state_logits=self.claim_state_head(latent_state),
            relation_type_logits=relation_type_logits,
            relation_target_logits=relation_target_logits,
            provenance_source_logits=provenance_source_logits,
            evidence_span_logits=evidence_span_logits,
            epistemic_status_logits=epistemic_status_logits,
            epistemic_values=epistemic_values,
            uncertainty_values=uncertainty_values,
            stability_logits=self.stability_head(latent_state),
            future_summary_latent=self.future_summary_head(latent_state),
            geometry_observable_values=self.geometry_head(latent_state),
            masks={
                "loss_masks": model_input.loss_masks,
                "availability_masks": model_input.availability_masks,
            },
            metadata={
                "emission_layer": "raw_emission",
                "hidden_oracle_repairs": False,
                "axc_out_schema_version": self.config.compatible_axc_out_spec_version,
            },
        )
