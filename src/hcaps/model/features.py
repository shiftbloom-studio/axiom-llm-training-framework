"""Numeric feature utilities shared by structured model modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from hcaps.model.config import AxiomModelConfig
from hcaps.model.types import ModelTensorGroup


def tensorize_group_records(
    records: list[dict[str, Any]],
    *,
    group_name: str,
    device: torch.device,
) -> ModelTensorGroup:
    """Tensorize one canonical group across records, padding ragged numeric lists."""

    group_values = [_record_group(record, group_name) for record in records]
    keys = sorted({key for group in group_values for key in group if _is_tensorizable(group[key])})
    tensor_group: ModelTensorGroup = {}
    for key in keys:
        values = [group.get(key, 0) for group in group_values]
        tensor_group[key] = values_to_tensor(values, device=device)
    return tensor_group


def values_to_tensor(values: Sequence[Any], *, device: torch.device) -> Tensor:
    normalized = [_numeric_sequence(value) for value in values]
    width = max((len(value) for value in normalized), default=1)
    width = max(width, 1)
    is_float = any(any(isinstance(item, float) for item in row) for row in normalized)
    dtype = torch.float32 if is_float else torch.long
    output = torch.zeros((len(normalized), width), dtype=dtype, device=device)
    for row_index, row in enumerate(normalized):
        if not row:
            continue
        row_tensor = torch.tensor(row[:width], dtype=dtype, device=device)
        output[row_index, : row_tensor.numel()] = row_tensor
    return output.squeeze(1) if width == 1 else output


def group_to_feature_tensor(
    group: ModelTensorGroup,
    *,
    batch_size: int,
    feature_count: int,
    device: torch.device,
) -> Tensor:
    """Convert a tensor group to bounded numeric features for dense modules."""

    features = torch.zeros((batch_size, feature_count), dtype=torch.float32, device=device)
    cursor = 0
    for key in sorted(group):
        if cursor >= feature_count:
            break
        tensor = _as_batch_matrix(group[key], batch_size=batch_size, device=device)
        width = min(tensor.shape[1], feature_count - cursor)
        features[:, cursor : cursor + width] = _scale_numeric(tensor[:, :width])
        cursor += width
    return features


def group_mask(group: ModelTensorGroup, *, batch_size: int, device: torch.device) -> Tensor:
    """Infer an item-level mask from explicit mask fields or tensor presence."""

    mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    for key, value in group.items():
        matrix = _as_batch_matrix(value, batch_size=batch_size, device=device)
        if key.endswith("_mask") or key.endswith("mask") or "mask" in key:
            mask |= matrix.ne(0).any(dim=1)
        else:
            mask |= matrix.ne(0).any(dim=1)
    return mask if group else torch.ones((batch_size,), dtype=torch.bool, device=device)


class GroupFeatureProjector(nn.Module):
    """Project one AXT tensor group into a model-dimensional context vector."""

    def __init__(self, config: AxiomModelConfig, *, output_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.output_dim = output_dim or config.model_dim
        self.projection = nn.Sequential(
            nn.Linear(config.numeric_feature_count, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        group: ModelTensorGroup,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        features = group_to_feature_tensor(
            group,
            batch_size=batch_size,
            feature_count=self.config.numeric_feature_count,
            device=device,
        )
        return cast(Tensor, self.projection(features))


def _record_group(record: dict[str, Any], group_name: str) -> dict[str, Any]:
    value = record.get(group_name, {})
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _is_tensorizable(value: Any) -> bool:
    if isinstance(value, bool | int | float):
        return True
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return all(_is_tensorizable(item) for item in value)
    return False


def _numeric_sequence(value: Any) -> list[int | float]:
    if isinstance(value, bool):
        return [int(value)]
    if isinstance(value, int | float):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        output: list[int | float] = []
        for item in value:
            output.extend(_numeric_sequence(item))
        return output
    return []


def _as_batch_matrix(value: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
    tensor = value.to(device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1).expand(batch_size, 1)
    elif tensor.ndim == 1:
        if tensor.shape[0] == batch_size:
            tensor = tensor.reshape(batch_size, 1)
        else:
            tensor = tensor.reshape(1, -1).expand(batch_size, -1)
    elif tensor.shape[0] != batch_size:
        tensor = tensor.reshape(1, -1).expand(batch_size, -1)
    else:
        tensor = tensor.reshape(batch_size, -1)
    return tensor.to(dtype=torch.float32)


def _scale_numeric(value: Tensor) -> Tensor:
    scaled = torch.sign(value) * torch.log1p(torch.abs(value)) / 32.0
    return torch.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=-1.0)
