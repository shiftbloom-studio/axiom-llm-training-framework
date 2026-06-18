"""Configuration for the Axiom structured-native model stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

GeometryMode = Literal["geometry_off", "geometry_features_from_axt", "geometry_provider_injected"]
RouterMode = Literal["router_off", "scalar_gate", "head_gate", "expert_gate", "loss_steering_only"]
CoreType = Literal["claim_field_transformer", "slot_mixer"]
ActivationName = Literal["gelu", "relu", "silu"]

P4_MODEL_SCHEMA_VERSION = "0.1.0"
COMPATIBLE_AXT_SPEC_VERSION = "1.0.0"
COMPATIBLE_AXC_OUT_SPEC_VERSION = "1.0.0"
MIN_CATEGORICAL_VOCAB_SIZE = 16


class AblationConfig(BaseModel):
    """Config-driven ablation switches for P4 model components."""

    model_config = ConfigDict(extra="forbid")

    text_only: bool = False
    structure_only_no_text_projection: bool = False
    no_provenance: bool = False
    no_relations: bool = False
    no_context: bool = False
    no_provider_context: bool = False
    no_side_channels: bool = False
    geometry_off: bool = False
    router_off: bool = False
    relation_neighborhood_off: bool = False

    @property
    def active(self) -> dict[str, bool]:
        return {
            "text_only": self.text_only,
            "structure_only_no_text_projection": self.structure_only_no_text_projection,
            "no_provenance": self.no_provenance,
            "no_relations": self.no_relations,
            "no_context": self.no_context,
            "no_provider_context": self.no_provider_context,
            "no_side_channels": self.no_side_channels,
            "geometry_off": self.geometry_off,
            "router_off": self.router_off,
            "relation_neighborhood_off": self.relation_neighborhood_off,
        }


class AxiomModelConfig(BaseModel):
    """P4 structured-native model configuration.

    The model stays structured at the boundary: AXT tensors enter as typed groups, AXC-out
    heads are emitted natively, and text projection remains a secondary comparison head.
    """

    model_config = ConfigDict(extra="forbid")

    model_dim: int = 64
    slot_dim: int = 64
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.0
    activation: ActivationName = "gelu"

    max_claim_slots: int = 16
    max_relation_neighbors: int = 8
    max_provenance_sources: int = 8
    max_negative_samples: int = 8
    max_text_length: int = 64
    text_vocab_size: int = 512
    categorical_vocab_size: int = 4096
    numeric_feature_count: int = 16
    geometry_observable_dim: int = 8

    axc_out_vocab_sizes: dict[str, int] = Field(
        default_factory=lambda: {
            "claim_state": 32,
            "relation_type": 64,
            "relation_target": 1024,
            "provenance_source": 1024,
            "evidence_span": 1024,
            "epistemic_status": 32,
            "stability": 8,
            "future_summary": 128,
        }
    )

    use_text_projection: bool = True
    use_epistemic_router: bool = True
    use_relation_conditioning: bool = True
    use_provenance_conditioning: bool = True
    use_provider_context: bool = True
    use_geometry_features: bool = False

    geometry_mode: GeometryMode = "geometry_off"
    router_mode: RouterMode = "scalar_gate"
    core_type: CoreType = "claim_field_transformer"
    parameter_budget_hint: str = "smoke"

    ablations: AblationConfig = Field(default_factory=AblationConfig)
    p4_model_schema_version: str = P4_MODEL_SCHEMA_VERSION
    compatible_axt_spec_version: str = COMPATIBLE_AXT_SPEC_VERSION
    compatible_axc_out_spec_version: str = COMPATIBLE_AXC_OUT_SPEC_VERSION

    @model_validator(mode="after")
    def validate_shapes(self) -> Self:
        if self.model_dim <= 0 or self.slot_dim <= 0:
            raise ValueError("model_dim and slot_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.text_vocab_size <= 1:
            raise ValueError("text_vocab_size must be greater than 1")
        if self.numeric_feature_count <= 0:
            raise ValueError("numeric_feature_count must be positive")
        if self.categorical_vocab_size <= MIN_CATEGORICAL_VOCAB_SIZE:
            raise ValueError("categorical_vocab_size must be greater than 16")
        for name, size in self.axc_out_vocab_sizes.items():
            if size <= 1:
                raise ValueError(f"AXC-out vocabulary {name!r} must be greater than 1")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"model config must be a YAML mapping: {path}")
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> Self:
        """Load either flat fields or the nested sections used by P4 configs."""

        flattened: dict[str, Any] = {}
        flattened.update({k: v for k, v in payload.items() if k not in _SECTION_NAMES})

        model_section = _dict_section(payload, "model")
        flattened.update(model_section)

        input_section = _dict_section(payload, "inputs")
        flattened.update(input_section)

        output_section = _dict_section(payload, "outputs")
        if "text_projection" in output_section:
            flattened["use_text_projection"] = bool(output_section.pop("text_projection"))
        if "axc_out_vocab_sizes" in output_section:
            flattened["axc_out_vocab_sizes"] = output_section.pop("axc_out_vocab_sizes")
        output_section.pop("axc_out", None)
        flattened.update(output_section)

        router_section = _dict_section(payload, "router")
        if "enabled" in router_section:
            flattened["use_epistemic_router"] = bool(router_section.pop("enabled"))
        if "mode" in router_section:
            flattened["router_mode"] = router_section.pop("mode")
        flattened.update(router_section)

        geometry_section = _dict_section(payload, "geometry")
        if "enabled" in geometry_section:
            flattened["use_geometry_features"] = bool(geometry_section.pop("enabled"))
        if "mode" in geometry_section:
            flattened["geometry_mode"] = geometry_section.pop("mode")
        flattened.update(geometry_section)

        ablations = _dict_section(payload, "ablations")
        if ablations:
            flattened["ablations"] = ablations

        return cls.model_validate(flattened)

    def effective_text_projection(self) -> bool:
        return self.use_text_projection and not self.ablations.structure_only_no_text_projection

    def effective_router_mode(self) -> RouterMode:
        if not self.use_epistemic_router or self.ablations.router_off:
            return "router_off"
        return self.router_mode

    def effective_geometry_mode(self) -> GeometryMode:
        if self.ablations.geometry_off or not self.use_geometry_features:
            return "geometry_off"
        return self.geometry_mode

    def enabled_module_flags(self) -> dict[str, bool]:
        return {
            "text_projection": self.effective_text_projection(),
            "epistemic_router": self.effective_router_mode() != "router_off",
            "relation_conditioning": self.use_relation_conditioning
            and not self.ablations.no_relations,
            "provenance_conditioning": self.use_provenance_conditioning
            and not self.ablations.no_provenance,
            "provider_context": self.use_provider_context
            and not self.ablations.no_provider_context
            and not self.ablations.no_side_channels,
            "geometry_features": self.effective_geometry_mode() != "geometry_off",
        }


_SECTION_NAMES = frozenset({"model", "inputs", "outputs", "router", "geometry", "ablations"})


def _dict_section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key!r} section must be a mapping")
    return dict(value)
