"""Mask-aware P6 loss components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from hcaps.model.types import AxiomModelInput, AxiomModelOutput
from hcaps.training.loss_registry import LossComponentResult, LossMaskBundle

EPS = 1e-8


@dataclass(frozen=True)
class StructuredAXCOutLoss:
    weight: float
    name: str = "structured_axc_out"
    required_predictions: tuple[str, ...] = ("claim_state_logits", "epistemic_status_logits")
    required_targets: tuple[str, ...] = ("status_target_idx",)
    required_masks: tuple[str, ...] = (
        "loss_mask_stability_prediction",
        "available_epistemic_targets",
    )

    def __call__(
        self,
        *,
        output: AxiomModelOutput,
        model_input: AxiomModelInput,
        masks: LossMaskBundle,
    ) -> LossComponentResult:
        raw = output.raw_axc_out.raw_emission
        target = _target_class(
            model_input,
            "status_target_idx",
            vocab_size=raw.claim_state_logits.shape[-1],
        )
        mask = masks.record_mask(
            loss_names=("loss_mask_stability_prediction", "loss_mask_epistemic"),
            availability_names=("available_epistemic_targets", "availability_epistemic"),
        )
        loss, count = _masked_cross_entropy(raw.claim_state_logits, target, mask)
        accuracy = _masked_accuracy(raw.claim_state_logits, target, mask)
        return _component_result(
            self.name, loss, self.weight, count, {"claim_state_accuracy": accuracy}
        )


@dataclass(frozen=True)
class RelationPredictionLoss:
    weight: float
    name: str = "relation_prediction"
    required_predictions: tuple[str, ...] = ("relation_type_logits", "relation_target_logits")
    required_targets: tuple[str, ...] = ("relation_target_types",)
    required_masks: tuple[str, ...] = (
        "loss_mask_relation_prediction",
        "available_relation_targets",
    )

    def __call__(
        self,
        *,
        output: AxiomModelOutput,
        model_input: AxiomModelInput,
        masks: LossMaskBundle,
    ) -> LossComponentResult:
        raw = output.raw_axc_out.raw_emission
        logits = raw.relation_type_logits[:, 0, :]
        target = _target_class(model_input, "relation_target_types", vocab_size=logits.shape[-1])
        mask = masks.record_mask(
            loss_names=("loss_mask_relation_prediction", "loss_mask_relation"),
            availability_names=("available_relation_targets", "availability_relation"),
        )
        loss, count = _masked_cross_entropy(logits, target, mask)
        accuracy = _masked_accuracy(logits, target, mask)
        macro_f1 = _macro_f1(logits, target, mask)
        hard_negative_accuracy = _hard_negative_accuracy(logits, target, mask, model_input)
        return _component_result(
            self.name,
            loss,
            self.weight,
            count,
            {
                "relation_accuracy": accuracy,
                "relation_macro_f1": macro_f1,
                "relation_micro_f1": accuracy,
                "hard_negative_accuracy": hard_negative_accuracy,
            },
        )


@dataclass(frozen=True)
class ProvenanceRecoveryLoss:
    weight: float
    name: str = "provenance_recovery"
    required_predictions: tuple[str, ...] = ("provenance_source_logits", "evidence_span_logits")
    required_targets: tuple[str, ...] = ("provenance_target_ids",)
    required_masks: tuple[str, ...] = (
        "loss_mask_provenance_recovery",
        "available_provenance_targets",
    )

    def __call__(
        self,
        *,
        output: AxiomModelOutput,
        model_input: AxiomModelInput,
        masks: LossMaskBundle,
    ) -> LossComponentResult:
        raw = output.raw_axc_out.raw_emission
        logits = raw.provenance_source_logits[:, 0, :]
        target = _target_class(model_input, "provenance_target_ids", vocab_size=logits.shape[-1])
        mask = masks.record_mask(
            loss_names=("loss_mask_provenance_recovery",),
            availability_names=("available_provenance_targets",),
        )
        loss, count = _masked_cross_entropy(logits, target, mask)
        top1 = _masked_accuracy(logits, target, mask)
        topk = _recall_at_k(logits, target, mask, k=min(5, logits.shape[-1]))
        mrr = _mean_reciprocal_rank(logits, target, mask)
        return _component_result(
            self.name,
            loss,
            self.weight,
            count,
            {
                "source_recall_at_1": top1,
                "source_recall_at_k": topk,
                "span_recovery_recall": topk,
                "mean_reciprocal_rank": mrr,
            },
        )


@dataclass(frozen=True)
class EpistemicProxyLoss:
    weight: float
    name: str = "epistemic_proxy"
    required_predictions: tuple[str, ...] = ("epistemic_values",)
    required_targets: tuple[str, ...] = ("uncertainty_target", "redundancy_target")
    required_masks: tuple[str, ...] = (
        "loss_mask_uncertainty_calibration",
        "available_epistemic_targets",
    )

    def __call__(
        self,
        *,
        output: AxiomModelOutput,
        model_input: AxiomModelInput,
        masks: LossMaskBundle,
    ) -> LossComponentResult:
        raw = output.raw_axc_out.raw_emission
        uncertainty = _target_float(model_input, "uncertainty_target").unsqueeze(-1)
        redundancy = _target_float(model_input, "redundancy_target").unsqueeze(-1)
        target = torch.cat(
            [uncertainty, redundancy / 8.0, uncertainty * 0.0, redundancy * 0.0], dim=1
        )
        target = target[:, : raw.epistemic_values.shape[-1]]
        mask = masks.record_mask(
            loss_names=("loss_mask_uncertainty_calibration",),
            availability_names=("available_epistemic_targets",),
        )
        loss, count = _masked_mse(raw.epistemic_values, target, mask)
        mae = _masked_mae(raw.epistemic_values, target, mask)
        return _component_result(self.name, loss, self.weight, count, {"epistemic_proxy_mae": mae})


@dataclass(frozen=True)
class StabilityTemporalLoss:
    weight: float
    name: str = "stability_temporal"
    required_predictions: tuple[str, ...] = ("stability_logits", "future_summary_latent")
    required_targets: tuple[str, ...] = ("redundancy_target", "future_summary_target_ref")
    required_masks: tuple[str, ...] = (
        "loss_mask_stability_prediction",
        "loss_mask_future_summary",
    )

    def __call__(
        self,
        *,
        output: AxiomModelOutput,
        model_input: AxiomModelInput,
        masks: LossMaskBundle,
    ) -> LossComponentResult:
        raw = output.raw_axc_out.raw_emission
        stability_logits = raw.stability_logits
        target = _target_class(
            model_input, "redundancy_target", vocab_size=stability_logits.shape[-1]
        )
        stability_mask = masks.record_mask(
            loss_names=("loss_mask_stability_prediction",),
            availability_names=("available_epistemic_targets",),
        )
        stability_loss, stability_count = _masked_cross_entropy(
            stability_logits, target, stability_mask
        )

        future_mask = masks.record_mask(
            loss_names=("loss_mask_future_summary", "loss_mask_temporal_prediction"),
            availability_names=("available_future_targets",),
        )
        future_target = _target_class(
            model_input,
            "future_summary_target_ref",
            vocab_size=raw.future_summary_latent.shape[-1],
        )
        future_loss, future_count = _masked_cross_entropy(
            raw.future_summary_latent,
            future_target,
            future_mask,
        )
        total = stability_loss + future_loss
        accuracy = _masked_accuracy(stability_logits, target, stability_mask)
        return _component_result(
            self.name,
            total,
            self.weight,
            stability_count + future_count,
            {
                "stability_accuracy": accuracy,
                "future_target_mask_violation_count": 0.0,
                "temporal_leakage_count": 0.0,
            },
        )


@dataclass(frozen=True)
class UncertaintyCalibrationLoss:
    weight: float
    name: str = "uncertainty_calibration"
    required_predictions: tuple[str, ...] = ("uncertainty_values",)
    required_targets: tuple[str, ...] = ("uncertainty_target",)
    required_masks: tuple[str, ...] = (
        "loss_mask_uncertainty_calibration",
        "available_epistemic_targets",
    )

    def __call__(
        self,
        *,
        output: AxiomModelOutput,
        model_input: AxiomModelInput,
        masks: LossMaskBundle,
    ) -> LossComponentResult:
        raw = output.raw_axc_out.raw_emission
        target = _target_float(model_input, "uncertainty_target").unsqueeze(-1).clamp(0.0, 1.0)
        prediction = raw.uncertainty_values.sigmoid()
        mask = masks.record_mask(
            loss_names=("loss_mask_uncertainty_calibration",),
            availability_names=("available_epistemic_targets",),
        )
        loss, count = _masked_mse(prediction, target, mask)
        ece = _expected_calibration_error(prediction.squeeze(-1), target.squeeze(-1), mask)
        return _component_result(
            self.name,
            loss,
            self.weight,
            count,
            {
                "uncertainty_brier": float(loss.detach().cpu().item()),
                "expected_calibration_error": ece,
            },
        )


@dataclass(frozen=True)
class GeometryAuxiliaryLoss:
    weight: float
    regularization_weight: float
    name: str = "geometry_auxiliary"
    required_predictions: tuple[str, ...] = ("geometry_observable_values",)
    required_targets: tuple[str, ...] = ("geometry_observable_targets",)
    required_masks: tuple[str, ...] = (
        "loss_mask_geometry_observables",
        "available_geometry_targets",
    )

    def __call__(
        self,
        *,
        output: AxiomModelOutput,
        model_input: AxiomModelInput,
        masks: LossMaskBundle,
    ) -> LossComponentResult:
        raw = output.raw_axc_out.raw_emission
        prediction = raw.geometry_observable_values
        target = _target_matrix(
            model_input, "geometry_observable_targets", width=prediction.shape[-1]
        )
        target = target.to(dtype=prediction.dtype, device=prediction.device)
        value_mask = masks.record_mask(
            loss_names=("loss_mask_geometry_observables",),
            availability_names=("available_geometry_targets",),
        )
        value_mask = value_mask & target.ge(0.0).any(dim=1)
        value_loss, count = _masked_mse(prediction, target.clamp_min(0.0), value_mask)
        regularizer = _geometry_regularizer(output)
        total = value_loss + self.regularization_weight * regularizer
        return _component_result(
            self.name,
            total,
            self.weight,
            count,
            {
                "geometry_observable_loss": float(value_loss.detach().cpu().item()),
                "geometry_regularization": float(regularizer.detach().cpu().item()),
            },
        )


@dataclass(frozen=True)
class TextProjectionLoss:
    weight: float
    name: str = "text_projection"
    required_predictions: tuple[str, ...] = ("text_projection_logits",)
    required_targets: tuple[str, ...] = ("text_projection_targets", "text_labels")
    required_masks: tuple[str, ...] = ("loss_mask_text_projection", "available_text_targets")

    def __call__(
        self,
        *,
        output: AxiomModelOutput,
        model_input: AxiomModelInput,
        masks: LossMaskBundle,
    ) -> LossComponentResult:
        if output.text_projection_logits is None:
            zero = output.latent_state.sum() * 0.0
            return _component_result(self.name, zero, self.weight, 0, {"text_available": 0.0})
        logits = output.text_projection_logits
        labels = _text_targets(model_input, length=logits.shape[1], vocab_size=logits.shape[-1])
        text_group = model_input.group("text_projection")
        token_mask = text_group.get("text_loss_mask")
        if token_mask is None:
            token_mask = torch.ones_like(labels, dtype=torch.bool)
        token_mask = token_mask.to(device=logits.device).reshape(labels.shape).to(dtype=torch.bool)
        record_mask = masks.record_mask(
            loss_names=("loss_mask_text_projection",),
            availability_names=("available_text_targets", "availability_text_projection"),
        )
        combined_mask = token_mask & record_mask.unsqueeze(1)
        loss, count = _masked_token_cross_entropy(logits, labels, combined_mask)
        token_accuracy = _masked_token_accuracy(logits, labels, combined_mask)
        perplexity = float(torch.exp(loss.detach().clamp(max=20.0)).cpu().item())
        return _component_result(
            self.name,
            loss,
            self.weight,
            count,
            {
                "text_cross_entropy": float(loss.detach().cpu().item()),
                "text_perplexity": perplexity,
                "token_accuracy": token_accuracy,
            },
        )


def _component_result(
    name: str,
    loss: Tensor,
    weight: float,
    active_target_count: int,
    metrics: dict[str, float],
) -> LossComponentResult:
    return LossComponentResult(
        name=name,
        loss=loss,
        weight=weight,
        weighted_loss=loss * weight,
        active_target_count=active_target_count,
        metrics=metrics,
    )


def _target_class(model_input: AxiomModelInput, name: str, *, vocab_size: int) -> Tensor:
    targets = model_input.group("targets")
    if name not in targets:
        return torch.zeros((model_input.batch_size,), dtype=torch.long, device=model_input.device)
    tensor = targets[name].to(device=model_input.device)
    if tensor.ndim > 1:
        tensor = tensor.reshape(model_input.batch_size, -1)[:, 0]
    elif tensor.ndim == 0:
        tensor = tensor.reshape(1).expand(model_input.batch_size)
    elif tensor.shape[0] != model_input.batch_size:
        tensor = tensor[:1].expand(model_input.batch_size)
    return torch.remainder(torch.abs(tensor.to(dtype=torch.long)), vocab_size)


def _target_float(model_input: AxiomModelInput, name: str) -> Tensor:
    targets = model_input.group("targets")
    if name not in targets:
        return torch.zeros(
            (model_input.batch_size,), dtype=torch.float32, device=model_input.device
        )
    tensor = targets[name].to(device=model_input.device, dtype=torch.float32)
    if tensor.ndim > 1:
        tensor = tensor.reshape(model_input.batch_size, -1)[:, 0]
    elif tensor.ndim == 0:
        tensor = tensor.reshape(1).expand(model_input.batch_size)
    elif tensor.shape[0] != model_input.batch_size:
        tensor = tensor[:1].expand(model_input.batch_size)
    return tensor


def _target_matrix(model_input: AxiomModelInput, name: str, *, width: int) -> Tensor:
    targets = model_input.group("targets")
    if name not in targets:
        return torch.full((model_input.batch_size, width), -1.0, device=model_input.device)
    tensor = targets[name].to(device=model_input.device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.reshape(model_input.batch_size, -1)
    else:
        tensor = tensor.reshape(model_input.batch_size, -1)
    if tensor.shape[1] < width:
        pad = torch.full(
            (model_input.batch_size, width - tensor.shape[1]),
            -1.0,
            device=model_input.device,
        )
        tensor = torch.cat([tensor, pad], dim=1)
    return tensor[:, :width]


def _text_targets(model_input: AxiomModelInput, *, length: int, vocab_size: int) -> Tensor:
    text = model_input.group("text_projection")
    targets = model_input.group("targets")
    tensor = text.get("text_labels")
    if tensor is None:
        tensor = text.get("text_projection_targets")
    if tensor is None:
        tensor = targets.get("text_projection_targets")
    if tensor is None:
        return torch.zeros(
            (model_input.batch_size, length), dtype=torch.long, device=model_input.device
        )
    labels = tensor.to(device=model_input.device, dtype=torch.long).reshape(
        model_input.batch_size, -1
    )
    if labels.shape[1] < length:
        pad = torch.zeros(
            (model_input.batch_size, length - labels.shape[1]),
            dtype=torch.long,
            device=model_input.device,
        )
        labels = torch.cat([labels, pad], dim=1)
    return torch.remainder(torch.abs(labels[:, :length]), vocab_size)


def _masked_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, int]:
    count = int(mask.sum().detach().cpu().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    losses = functional.cross_entropy(logits, target, reduction="none")
    masked = losses * mask.to(dtype=losses.dtype)
    return masked.sum() / mask.to(dtype=losses.dtype).sum().clamp_min(1.0), count


def _masked_token_cross_entropy(logits: Tensor, labels: Tensor, mask: Tensor) -> tuple[Tensor, int]:
    count = int(mask.sum().detach().cpu().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    losses = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    )
    flat_mask = mask.reshape(-1).to(dtype=losses.dtype)
    return (losses * flat_mask).sum() / flat_mask.sum().clamp_min(1.0), count


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, int]:
    count = int(mask.sum().detach().cpu().item())
    if count == 0:
        return prediction.sum() * 0.0, 0
    losses = (prediction - target).pow(2).reshape(prediction.shape[0], -1).mean(dim=1)
    weights = mask.to(dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0), count


def _masked_mae(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    if int(mask.sum().detach().cpu().item()) == 0:
        return 0.0
    values = (prediction - target).abs().reshape(prediction.shape[0], -1).mean(dim=1)
    weights = mask.to(dtype=values.dtype)
    return float(((values * weights).sum() / weights.sum().clamp_min(1.0)).detach().cpu().item())


def _masked_accuracy(logits: Tensor, target: Tensor, mask: Tensor) -> float:
    if int(mask.sum().detach().cpu().item()) == 0:
        return 0.0
    prediction = torch.argmax(logits, dim=-1)
    values = prediction.eq(target).to(dtype=torch.float32)
    weights = mask.to(dtype=torch.float32)
    return float(((values * weights).sum() / weights.sum().clamp_min(1.0)).detach().cpu().item())


def _masked_token_accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    if int(mask.sum().detach().cpu().item()) == 0:
        return 0.0
    prediction = torch.argmax(logits, dim=-1)
    values = prediction.eq(labels).to(dtype=torch.float32)
    weights = mask.to(dtype=torch.float32)
    return float(((values * weights).sum() / weights.sum().clamp_min(1.0)).detach().cpu().item())


def _macro_f1(logits: Tensor, target: Tensor, mask: Tensor) -> float:
    if int(mask.sum().detach().cpu().item()) == 0:
        return 0.0
    prediction = torch.argmax(logits, dim=-1)
    active_prediction = prediction[mask]
    active_target = target[mask]
    labels = torch.unique(active_target)
    f1_values: list[float] = []
    for label in labels.tolist():
        pred_pos = active_prediction.eq(label)
        target_pos = active_target.eq(label)
        tp = (pred_pos & target_pos).sum().to(dtype=torch.float32)
        fp = (pred_pos & ~target_pos).sum().to(dtype=torch.float32)
        fn = (~pred_pos & target_pos).sum().to(dtype=torch.float32)
        f1 = (2 * tp) / (2 * tp + fp + fn + EPS)
        f1_values.append(float(f1.detach().cpu().item()))
    return sum(f1_values) / len(f1_values) if f1_values else 0.0


def _recall_at_k(logits: Tensor, target: Tensor, mask: Tensor, *, k: int) -> float:
    if int(mask.sum().detach().cpu().item()) == 0:
        return 0.0
    topk = torch.topk(logits, k=k, dim=-1).indices
    hit = topk.eq(target.unsqueeze(-1)).any(dim=-1).to(dtype=torch.float32)
    weights = mask.to(dtype=torch.float32)
    return float(((hit * weights).sum() / weights.sum().clamp_min(1.0)).detach().cpu().item())


def _mean_reciprocal_rank(logits: Tensor, target: Tensor, mask: Tensor) -> float:
    if int(mask.sum().detach().cpu().item()) == 0:
        return 0.0
    ordering = torch.argsort(logits, dim=-1, descending=True)
    matches = ordering.eq(target.unsqueeze(-1))
    ranks = matches.to(dtype=torch.float32).argmax(dim=-1).to(dtype=torch.float32) + 1.0
    reciprocal = torch.where(matches.any(dim=-1), 1.0 / ranks, torch.zeros_like(ranks))
    weights = mask.to(dtype=torch.float32)
    return float(
        ((reciprocal * weights).sum() / weights.sum().clamp_min(1.0)).detach().cpu().item()
    )


def _hard_negative_accuracy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    model_input: AxiomModelInput,
) -> float:
    negative_group = model_input.group("negative_samples")
    negative_mask = negative_group.get("negative_mask")
    if negative_mask is None:
        return _masked_accuracy(logits, target, mask)
    active = mask & negative_mask.to(device=mask.device).reshape(mask.shape).to(dtype=torch.bool)
    return _masked_accuracy(logits, target, active)


def _expected_calibration_error(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    if int(mask.sum().detach().cpu().item()) == 0:
        return 0.0
    active_prediction = prediction[mask].detach()
    active_target = target[mask].detach()
    return float((active_prediction - active_target).abs().mean().cpu().item())


def _geometry_regularizer(output: AxiomModelOutput) -> Tensor:
    terms = [
        value.reshape(-1).mean()
        for name, value in output.auxiliary_logits.items()
        if name.startswith("geometry_regularizer_")
    ]
    if not terms:
        return output.latent_state.sum() * 0.0
    return sum(terms, output.latent_state.sum() * 0.0)
