"""Interpreter boundary for AXC-out raw emissions."""

from __future__ import annotations

from typing import Any

import torch

from hcaps.model.axc_out import AXCOutEmission, AXCOutRawEmission
from hcaps.model.types import FORBIDDEN_OUTPUT_FIELD_FRAGMENTS

MIN_LOGIT_RANK = 2


class AXCOutInterpreterBoundary:
    """Validate and summarize raw AXC-out without repairing model emissions."""

    def __call__(
        self,
        raw_emission: AXCOutRawEmission,
        *,
        text_projection_shape: tuple[int, ...] | None,
    ) -> AXCOutEmission:
        forbidden = _forbidden_field_names(raw_emission)
        shape_summary = raw_emission.head_shapes()
        batch_sizes = {shape[0] for shape in shape_summary.values() if shape}
        valid_shape = len(batch_sizes) == 1
        validated = {
            "valid_shape": valid_shape,
            "valid_mask_alignment": "loss_masks" in raw_emission.masks
            and "availability_masks" in raw_emission.masks,
            "invalid_missing_required_head": _missing_required_heads(raw_emission),
            "invalid_forbidden_truth_field_name": forbidden,
            "hidden_oracle_repairs": False,
            "raw_emission_preserved": True,
        }
        interpreted = {
            "model_emitted": True,
            "top_candidates": _top_candidates(raw_emission),
            "confidence_source": "raw model logits",
            "repair_policy": "none",
        }
        text_projection = {
            "available": text_projection_shape is not None,
            "logits_shape": text_projection_shape,
            "role": "secondary_projection_and_comparison_interface",
        }
        return AXCOutEmission(
            raw_emission=raw_emission,
            validated_axc_out=validated,
            interpreted_projection=interpreted,
            text_projection=text_projection,
        )


def _missing_required_heads(raw_emission: AXCOutRawEmission) -> list[str]:
    return [name for name, tensor in raw_emission.tensor_fields().items() if tensor.numel() == 0]


def _forbidden_field_names(raw_emission: AXCOutRawEmission) -> list[str]:
    names = raw_emission.tensor_fields().keys()
    return [
        name
        for name in names
        if any(fragment in name.lower() for fragment in FORBIDDEN_OUTPUT_FIELD_FRAGMENTS)
    ]


def _top_candidates(raw_emission: AXCOutRawEmission) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    with torch.no_grad():
        for name, tensor in raw_emission.tensor_fields().items():
            if tensor.ndim < MIN_LOGIT_RANK or tensor.shape[-1] < MIN_LOGIT_RANK:
                continue
            values, indices = torch.topk(tensor.reshape(tensor.shape[0], -1), k=1, dim=-1)
            candidates[name] = {
                "index": indices.squeeze(-1).detach().cpu().tolist(),
                "score": values.squeeze(-1).detach().cpu().tolist(),
            }
    return candidates
