"""Configuration for learned claim-field geometry."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GeometryMode(StrEnum):
    OFF = "off"
    PRECOMPUTED = "precomputed"
    LEARNED = "learned"
    REFERENCE = "reference"


class TransportOperator(StrEnum):
    MATRIX_EXP = "matrix_exp"
    CAYLEY = "cayley"


class RegularizerConfig(BaseModel):
    """Weights are carried for P6 configs; P5 returns unweighted terms."""

    model_config = ConfigDict(extra="forbid")

    connection_norm: float = 0.0
    curvature_energy: float = 0.0
    path_consistency: float = 0.0
    context_smoothness: float = 0.0
    transport_identity_bias: float = 0.0
    non_degenerate_usage: float = 0.0


class GeometryExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_raw_matrices: bool = False


class GeometryConfig(BaseModel):
    """P5 geometry module configuration."""

    model_config = ConfigDict(extra="forbid")

    mode: GeometryMode = GeometryMode.LEARNED
    fiber_dim: int = 16
    geometry_feature_dim: int = 64
    connection_rank: int | None = None
    connection_hidden: int = 0
    connection_scale: float = 1.0
    connection_max_norm: float = 4.0
    max_transition_types: int = 16
    max_relation_types: int = 128
    max_node_types: int = 16
    transport_operator: TransportOperator = TransportOperator.MATRIX_EXP
    max_loop_length: int = 4
    max_loops_per_batch: int = 128
    max_loops_per_claim_family: int = 8
    max_context_degree: int = 16
    seed: int = 13
    use_provider_context: bool = True
    use_provider_disagreement_edges: bool = True
    use_relation_type_conditioning: bool = True
    provider_shuffle_control_ready: bool = True
    router_use_geometry: bool = True
    router_geometry_gate_init: float = 0.05
    router_geometry_dropout: float = 0.0
    parameter_matched: bool = False
    regularizers: RegularizerConfig = Field(default_factory=RegularizerConfig)
    export: GeometryExportConfig = Field(default_factory=GeometryExportConfig)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        if self.fiber_dim <= 1:
            raise ValueError("fiber_dim must be greater than 1")
        if self.geometry_feature_dim <= 0:
            raise ValueError("geometry_feature_dim must be positive")
        if self.max_loop_length <= 0:
            raise ValueError("max_loop_length must be positive")
        if self.max_loops_per_batch < 0:
            raise ValueError("max_loops_per_batch cannot be negative")
        if self.max_context_degree <= 0:
            raise ValueError("max_context_degree must be positive")
        if not 0.0 <= self.router_geometry_dropout < 1.0:
            raise ValueError("router_geometry_dropout must be in [0, 1)")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"geometry config must be a YAML mapping: {path}")
        return cls.model_validate(payload)

    def mode_name(self) -> Literal["off", "precomputed", "learned", "reference"]:
        return self.mode.value


DEFAULT_CONTEXT_TRANSITION_TYPES: tuple[str, ...] = (
    "same_claim_cross_domain",
    "same_claim_cross_provider",
    "same_claim_cross_source_type",
    "same_claim_cross_method",
    "same_claim_cross_community",
    "same_claim_cross_language",
    "same_claim_cross_benchmark_family",
    "provider_disagreement_transition",
    "manual_reference_transition",
    "unknown_transition",
)


def config_summary(config: GeometryConfig) -> dict[str, Any]:
    return {
        "mode": config.mode.value,
        "fiber_dim": config.fiber_dim,
        "geometry_feature_dim": config.geometry_feature_dim,
        "transport_operator": config.transport_operator.value,
        "max_loop_length": config.max_loop_length,
        "max_loops_per_batch": config.max_loops_per_batch,
        "use_provider_context": config.use_provider_context,
        "use_provider_disagreement_edges": config.use_provider_disagreement_edges,
        "export_include_raw_matrices": config.export.include_raw_matrices,
    }
