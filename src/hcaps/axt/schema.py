"""Shared AXT runtime schema constants and light data structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from hcaps.format.hashing import sha256_text

AXT_FORMAT = "AXT"
AXT_FORMAT_VERSION = "1.0.0"
AXT_COMPILER_VERSION = "0.1.0"

type TensorArray = np.ndarray
type TensorGroup = dict[str, TensorArray]
type TensorGroups = dict[str, TensorGroup]
type JsonObject = dict[str, object]

REQUIRED_TENSOR_GROUPS: tuple[str, ...] = (
    "ids",
    "claim",
    "temporal",
    "lateral_context",
    "provider_context",
    "epistemic_state",
    "provenance",
    "relations",
    "relation_neighborhoods",
    "negative_samples",
    "geometry_observables",
    "text_projection",
    "targets",
    "availability_masks",
    "loss_masks",
    "split_masks",
    "metadata",
)

PREDICTOR_VISIBLE_GROUPS: tuple[str, ...] = (
    "ids",
    "claim",
    "temporal",
    "lateral_context",
    "provider_context",
    "epistemic_state",
    "provenance",
    "relations",
    "relation_neighborhoods",
    "negative_samples",
    "geometry_observables",
    "text_projection",
)

TARGET_GROUPS: tuple[str, ...] = (
    "targets",
    "availability_masks",
    "loss_masks",
)

LOSS_MASK_NAMES: tuple[str, ...] = (
    "loss_mask_text_projection",
    "loss_mask_relation_prediction",
    "loss_mask_provenance_recovery",
    "loss_mask_stability_prediction",
    "loss_mask_uncertainty_calibration",
    "loss_mask_future_summary",
    "loss_mask_temporal_prediction",
    "loss_mask_geometry_observables",
    "loss_mask_provider_context",
    "loss_mask_negative_sampling",
)

AVAILABILITY_MASK_NAMES: tuple[str, ...] = (
    "available_relation_targets",
    "available_provenance_targets",
    "available_epistemic_targets",
    "available_future_targets",
    "available_geometry_targets",
    "available_text_targets",
    "available_evaluation_references",
)

TEXT_RENDER_MODES: tuple[str, ...] = (
    "flat_text",
    "structured_text",
    "capsule_text",
)

SPECIAL_TOKENS: tuple[str, ...] = (
    "<AX_CLAIM>",
    "<AX_PROVENANCE>",
    "<AX_RELATION>",
    "<AX_TEMPORAL>",
    "<AX_CONTEXT>",
    "<AX_EPISTEMIC>",
    "<AX_GEOMETRY>",
    "<AX_VIEW>",
    "<AX_END>",
)

MISSING_INT = -1
MISSING_FLOAT = -1.0


@dataclass(frozen=True)
class LoadedAxtInput:
    """Resolved AXC/AXP input records and optional P1 side artifacts."""

    input_path: Path
    source_format: Literal["AXC", "AXP"]
    capsules_path: Path
    capsules: list[Any]
    split_ids: set[str] | None
    provider_traces: list[dict[str, object]]
    negative_pools: list[dict[str, object]]
    evaluation_references: list[dict[str, object]]
    source_registry: list[dict[str, object]]
    claim_families: list[dict[str, object]]
    package_reports: dict[str, object]


def stable_int64(*parts: object) -> int:
    """Return a deterministic positive int64 reference for string/hash fields."""

    digest = sha256_text(*parts)
    return int(digest[:15], 16)


def timestamp_seconds(value: datetime | None) -> int:
    """Represent datetimes as Unix seconds, with -1 for missing values."""

    if value is None:
        return MISSING_INT
    return int(value.timestamp())


def as_i64(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def as_f32(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)
