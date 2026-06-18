"""Typed runtime objects for the structured-native model stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

ModelTensorGroup = dict[str, Tensor]

SLOT_TYPE_IDS: dict[str, int] = {
    "claim_identity": 0,
    "claim_state": 1,
    "epistemic_state": 2,
    "temporal_scope": 3,
    "lateral_context": 4,
    "provider_context": 5,
    "provenance": 6,
    "relations": 7,
    "relation_neighborhoods": 8,
    "negative_samples": 9,
    "geometry_features": 10,
    "text_projection": 11,
}

FORBIDDEN_OUTPUT_FIELD_FRAGMENTS: tuple[str, ...] = (
    "truth",
    "is_true",
    "correct",
    "is_correct",
    "ground_truth",
    "proven_true",
)


@dataclass(frozen=True)
class AxiomModelInput:
    """Normalized model-facing view of an AXT runtime batch."""

    groups: dict[str, ModelTensorGroup]
    loss_masks: ModelTensorGroup
    availability_masks: ModelTensorGroup
    metadata: dict[str, Any]
    batch_size: int
    device: torch.device

    def group(self, name: str) -> ModelTensorGroup:
        return self.groups.get(name, {})


@dataclass(frozen=True)
class TypedSlotBundle:
    """Typed slot tensor bundle preserving source groups through the stack."""

    slot_tensor: Tensor
    slot_type_ids: Tensor
    slot_mask: Tensor
    source_groups: tuple[str, ...]
    group_ranges: dict[str, tuple[int, int]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class EncoderOutput:
    """Structured encoder output with group vectors still separated."""

    slot_states: Tensor
    slot_type_ids: Tensor
    slot_mask: Tensor
    source_groups: tuple[str, ...]
    claim_state_vector: Tensor
    epistemic_vector: Tensor
    relation_context_vector: Tensor
    provenance_context_vector: Tensor
    temporal_vector: Tensor
    lateral_context_vector: Tensor
    provider_context_vector: Tensor
    geometry_context_vector: Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CoreOutput:
    """Full-complexity core output before routing and decoding."""

    slot_states: Tensor
    slot_type_ids: Tensor
    slot_mask: Tensor
    source_groups: tuple[str, ...]
    claim_state_vector: Tensor
    relation_state: Tensor
    provenance_state: Tensor
    context_state: Tensor
    router_state: Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class AxiomModelOutput:
    """Loss-ready P4 forward output.

    P4 emits raw structured AXC-out tensors and text-projection logits, but it does not
    compute final P6 losses or make evaluation claims.
    """

    latent_state: Tensor
    slot_states: Tensor
    raw_axc_out: Any
    text_projection_logits: Tensor | None
    auxiliary_logits: dict[str, Tensor]
    router_diagnostics: dict[str, Any]
    geometry_diagnostics: dict[str, Any]
    masks: dict[str, ModelTensorGroup]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]
