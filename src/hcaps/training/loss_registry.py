"""Modular P6 loss registry with mask-aware aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor

from hcaps.model.types import AxiomModelInput, AxiomModelOutput, ModelTensorGroup
from hcaps.training.config import LossWeights


@dataclass(frozen=True)
class LossComponentResult:
    name: str
    loss: Tensor
    weight: float
    weighted_loss: Tensor
    active_target_count: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LossResult:
    total_loss: Tensor
    components: dict[str, LossComponentResult]

    def metrics(self) -> dict[str, float]:
        output = {"loss_total": float(self.total_loss.detach().cpu().item())}
        for name, component in self.components.items():
            output[f"loss_{name}"] = float(component.loss.detach().cpu().item())
            output[f"weighted_loss_{name}"] = float(component.weighted_loss.detach().cpu().item())
            output[f"active_targets_{name}"] = float(component.active_target_count)
            output.update({f"{name}_{key}": value for key, value in component.metrics.items()})
        return output

    def active_targets(self) -> dict[str, int]:
        return {name: component.active_target_count for name, component in self.components.items()}


@dataclass(frozen=True)
class LossMaskBundle:
    loss_masks: ModelTensorGroup
    availability_masks: ModelTensorGroup
    batch_size: int
    device: torch.device

    def record_mask(
        self,
        *,
        loss_names: tuple[str, ...],
        availability_names: tuple[str, ...],
    ) -> Tensor:
        loss_mask = _first_mask(
            self.loss_masks,
            loss_names,
            batch_size=self.batch_size,
            device=self.device,
            required=True,
        )
        availability_mask = _first_mask(
            self.availability_masks,
            availability_names,
            batch_size=self.batch_size,
            device=self.device,
            required=True,
        )
        return loss_mask & availability_mask


class LossComponent(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def weight(self) -> float: ...

    @property
    def required_predictions(self) -> tuple[str, ...]: ...

    @property
    def required_targets(self) -> tuple[str, ...]: ...

    @property
    def required_masks(self) -> tuple[str, ...]: ...

    def __call__(
        self,
        *,
        output: AxiomModelOutput,
        model_input: AxiomModelInput,
        masks: LossMaskBundle,
    ) -> LossComponentResult:
        """Compute one weighted loss component."""


class LossRegistry:
    """Aggregate declared loss components with explicit weights."""

    def __init__(self, components: list[LossComponent]) -> None:
        self.components = components

    def compute(self, *, output: AxiomModelOutput, model_input: AxiomModelInput) -> LossResult:
        masks = LossMaskBundle(
            loss_masks=model_input.loss_masks,
            availability_masks=model_input.availability_masks,
            batch_size=model_input.batch_size,
            device=model_input.device,
        )
        component_results = {
            component.name: component(output=output, model_input=model_input, masks=masks)
            for component in self.components
        }
        if component_results:
            total = sum(
                (component.weighted_loss for component in component_results.values()),
                output.latent_state.sum() * 0.0,
            )
        else:
            total = output.latent_state.sum() * 0.0
        return LossResult(total_loss=total, components=component_results)


def build_default_loss_registry(weights: LossWeights) -> LossRegistry:
    """Build the full P6 structured + text loss registry."""

    from hcaps.training.losses import (  # noqa: PLC0415
        EpistemicProxyLoss,
        GeometryAuxiliaryLoss,
        ProvenanceRecoveryLoss,
        RelationPredictionLoss,
        StabilityTemporalLoss,
        StructuredAXCOutLoss,
        TextProjectionLoss,
        UncertaintyCalibrationLoss,
    )

    return LossRegistry(
        [
            StructuredAXCOutLoss(weights.structured_axc_out),
            RelationPredictionLoss(weights.relation_prediction),
            ProvenanceRecoveryLoss(weights.provenance_recovery),
            EpistemicProxyLoss(weights.epistemic_proxy),
            StabilityTemporalLoss(weights.stability_temporal),
            UncertaintyCalibrationLoss(weights.uncertainty_calibration),
            GeometryAuxiliaryLoss(weights.geometry_observables, weights.geometry_regularization),
            TextProjectionLoss(weights.text_projection),
        ]
    )


def _first_mask(
    group: ModelTensorGroup,
    names: tuple[str, ...],
    *,
    batch_size: int,
    device: torch.device,
    required: bool,
) -> Tensor:
    for name in names:
        if name in group:
            tensor = group[name].to(device=device)
            if tensor.ndim == 0:
                tensor = tensor.reshape(1).expand(batch_size)
            elif tensor.ndim > 1:
                tensor = tensor.reshape(batch_size, -1).any(dim=1)
            elif tensor.shape[0] != batch_size:
                tensor = tensor[:1].expand(batch_size)
            return tensor.to(dtype=torch.bool)
    if required:
        joined = ", ".join(names)
        raise ValueError(f"required P6 loss/availability mask is missing: {joined}")
    return torch.zeros((batch_size,), dtype=torch.bool, device=device)
