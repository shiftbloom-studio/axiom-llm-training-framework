"""Structured diagnostics for P4 model forwards."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from hcaps.model.axc_out import AXCOutEmission
from hcaps.model.config import AxiomModelConfig
from hcaps.model.parameter_count import parameter_count
from hcaps.model.types import AxiomModelInput, TypedSlotBundle


def model_diagnostics(
    *,
    config: AxiomModelConfig,
    model: nn.Module,
    model_input: AxiomModelInput,
    slots: TypedSlotBundle,
    axc_out: AXCOutEmission,
    text_projection_logits: Tensor | None,
    router_summary: dict[str, Any],
    geometry_summary: dict[str, Any],
    stage_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "parameter_count": parameter_count(model),
        "enabled_modules": config.enabled_module_flags(),
        "active_ablation_modes": config.ablations.active,
        "input_shape_summary": model_input.metadata.get("input_shape_summary", {}),
        "slot_count_by_type": slot_count_by_type(slots),
        "mask_summary": mask_summary(model_input),
        "router_summary": router_summary,
        "geometry_summary": geometry_summary,
        "output_shape_summary": {
            "text_projection_shape": tuple(text_projection_logits.shape)
            if text_projection_logits is not None
            else None,
            "axc_out_head_shapes": axc_out.head_shapes(),
        },
        "stage_diagnostics": stage_diagnostics,
    }


def slot_count_by_type(slots: TypedSlotBundle) -> dict[str, int]:
    return {name: slots.source_groups.count(name) for name in sorted(set(slots.source_groups))}


def mask_summary(model_input: AxiomModelInput) -> dict[str, Any]:
    return {
        "loss_masks": {
            key: _mask_count(value) for key, value in sorted(model_input.loss_masks.items())
        },
        "availability_masks": {
            key: _mask_count(value) for key, value in sorted(model_input.availability_masks.items())
        },
    }


def _mask_count(value: Tensor) -> int:
    tensor = value.detach()
    if tensor.dtype != torch.bool:
        tensor = tensor.ne(0)
    return int(tensor.sum().item())
