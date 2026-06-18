"""AXC-out raw emission objects for P4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class AXCOutRawEmission:
    """Native structured model emission before interpreter validation."""

    claim_state_logits: Tensor
    relation_type_logits: Tensor
    relation_target_logits: Tensor
    provenance_source_logits: Tensor
    evidence_span_logits: Tensor
    epistemic_status_logits: Tensor
    epistemic_values: Tensor
    uncertainty_values: Tensor
    stability_logits: Tensor
    future_summary_latent: Tensor
    geometry_observable_values: Tensor
    masks: dict[str, dict[str, Tensor]]
    metadata: dict[str, Any]

    def tensor_fields(self) -> dict[str, Tensor]:
        return {
            "claim_state_logits": self.claim_state_logits,
            "relation_type_logits": self.relation_type_logits,
            "relation_target_logits": self.relation_target_logits,
            "provenance_source_logits": self.provenance_source_logits,
            "evidence_span_logits": self.evidence_span_logits,
            "epistemic_status_logits": self.epistemic_status_logits,
            "epistemic_values": self.epistemic_values,
            "uncertainty_values": self.uncertainty_values,
            "stability_logits": self.stability_logits,
            "future_summary_latent": self.future_summary_latent,
            "geometry_observable_values": self.geometry_observable_values,
        }

    def head_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(tensor.shape) for name, tensor in self.tensor_fields().items()}


@dataclass(frozen=True)
class AXCOutEmission:
    """Boundary object preserving raw, validated, and interpreted layers."""

    raw_emission: AXCOutRawEmission
    validated_axc_out: dict[str, Any]
    interpreted_projection: dict[str, Any]
    text_projection: dict[str, Any]

    def head_shapes(self) -> dict[str, tuple[int, ...]]:
        return self.raw_emission.head_shapes()
