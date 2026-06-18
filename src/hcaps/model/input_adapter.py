"""Structured AXT input adapter for the P4 model stack."""

from __future__ import annotations

from typing import Any

import torch

from hcaps.model.batch import records_from_batch
from hcaps.model.config import AxiomModelConfig
from hcaps.model.features import tensorize_group_records
from hcaps.model.types import AxiomModelInput

CANONICAL_GROUPS: tuple[str, ...] = (
    "ids",
    "claim",
    "text_projection",
    "temporal",
    "lateral_context",
    "provider_context",
    "epistemic_state",
    "provenance",
    "relations",
    "relation_neighborhoods",
    "negative_samples",
    "geometry_observables",
    "targets",
    "availability_masks",
    "loss_masks",
)

GROUP_ALIASES: dict[str, str] = {
    "epistemic": "epistemic_state",
    "neighborhood": "relation_neighborhoods",
    "geometry_slots": "geometry_observables",
}


class StructuredInputAdapter:
    """Convert P3 AXT runtime batches into typed tensor groups.

    The adapter consumes AXT records and groups only. It does not parse AXC, call providers,
    infer hidden targets, or collapse the batch to capsule text.
    """

    def __init__(
        self,
        config: AxiomModelConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device or "cpu")

    def __call__(self, batch: object) -> AxiomModelInput:
        if isinstance(batch, AxiomModelInput):
            return batch
        records = [_canonicalize_record(record) for record in records_from_batch(batch)]
        if not records:
            raise ValueError("AXT model batch must contain at least one record")

        batch_size = len(records)
        groups = {
            group_name: tensorize_group_records(records, group_name=group_name, device=self.device)
            for group_name in CANONICAL_GROUPS
            if group_name not in {"loss_masks", "availability_masks"}
        }
        loss_masks = tensorize_group_records(records, group_name="loss_masks", device=self.device)
        availability_masks = tensorize_group_records(
            records,
            group_name="availability_masks",
            device=self.device,
        )

        metadata = {
            "records": [record.get("metadata", {}) for record in records],
            "input_shape_summary": _shape_summary(groups),
            "canonical_groups": tuple(groups),
        }
        return AxiomModelInput(
            groups=groups,
            loss_masks=loss_masks,
            availability_masks=availability_masks,
            metadata=metadata,
            batch_size=batch_size,
            device=self.device,
        )


def _canonicalize_record(record: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    for old_name, new_name in GROUP_ALIASES.items():
        if old_name in output and new_name not in output:
            output[new_name] = output[old_name]
    for group_name in CANONICAL_GROUPS:
        output.setdefault(group_name, {})
    return output


def _shape_summary(
    groups: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, tuple[int, ...]]]:
    return {
        group_name: {field_name: tuple(tensor.shape) for field_name, tensor in group.items()}
        for group_name, group in groups.items()
    }
