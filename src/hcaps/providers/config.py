"""Provider ingress configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from hcaps.extraction.contracts import ProviderMode

CacheMode = Literal["live", "cache_only", "refresh"]
ProviderType = Literal["deterministic", "openai_compatible", "python_callable"]


class ProviderEndpointConfig(BaseModel):
    """Configuration for one extraction provider endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    type: ProviderType
    provider_family: str = "deterministic"
    provider_mode: ProviderMode = "deterministic"
    model: str = "deterministic-bootstrap-v0.2"
    base_url: str | None = None
    api_key_env: str | None = None
    callable_path: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    cache_mode: CacheMode = "live"
    dry_run: bool = False

    @model_validator(mode="after")
    def require_type_specific_fields(self) -> ProviderEndpointConfig:
        if self.type == "openai_compatible" and not self.base_url:
            msg = "openai_compatible providers require base_url"
            raise ValueError(msg)
        if self.type == "python_callable" and not self.callable_path:
            msg = "python_callable providers require callable_path"
            raise ValueError(msg)
        return self


class ProviderCacheConfig(BaseModel):
    """Cache root and default mode for provider replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root_path: Path = Path(".cache/axiom/providers")
    mode: CacheMode = "live"

    @field_serializer("root_path")
    def serialize_path(self, value: Path) -> str:
        return str(value)


class ProviderGateConfig(BaseModel):
    """Deterministic escalation-gate settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    min_confidence: float = Field(default=0.72, ge=0.0, le=1.0)
    high_impact_claim_types: list[str] = Field(default_factory=list)
    max_escalation_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    escalate_on_warnings: bool = True


class ProviderCascadeConfig(BaseModel):
    """Config-driven local-first provider cascade."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    gate: ProviderGateConfig = Field(default_factory=ProviderGateConfig)
    merge_policy: str = "schema_grounded_confidence_priority_v0.1"


class ProviderIngressConfig(BaseModel):
    """Full provider ingress configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary: ProviderEndpointConfig
    escalation: ProviderEndpointConfig | None = None
    cascade: ProviderCascadeConfig = Field(default_factory=ProviderCascadeConfig)
    cache: ProviderCacheConfig = Field(default_factory=ProviderCacheConfig)


def load_provider_ingress_config(path: str | Path) -> ProviderIngressConfig:
    """Load provider ingress config from a YAML file."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        msg = "provider config must be a mapping"
        raise ValueError(msg)
    provider_payload = payload.get("providers", payload)
    if not isinstance(provider_payload, dict):
        msg = "provider config 'providers' must be a mapping"
        raise ValueError(msg)
    normalized = _normalize_provider_payload(provider_payload, Path(path).parent)
    return ProviderIngressConfig.model_validate(normalized)


def _normalize_provider_payload(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    normalized = dict(payload)
    cache = normalized.get("cache")
    if isinstance(cache, dict) and isinstance(cache.get("root_path"), str):
        root_path = Path(cache["root_path"])
        if not root_path.is_absolute():
            cache = dict(cache)
            cache["root_path"] = (base_dir / root_path).resolve()
            normalized["cache"] = cache
    if "primary" in normalized and isinstance(normalized["primary"], dict):
        normalized["primary"] = _normalize_endpoint(normalized["primary"], "primary")
    if "escalation" in normalized and isinstance(normalized["escalation"], dict):
        normalized["escalation"] = _normalize_endpoint(normalized["escalation"], "escalation")
    return normalized


def _normalize_endpoint(payload: dict[str, Any], default_id: str) -> dict[str, Any]:
    endpoint = dict(payload)
    endpoint.setdefault("provider_id", default_id)
    endpoint_type = endpoint.get("type")
    if endpoint_type == "deterministic":
        endpoint.setdefault("provider_family", "deterministic")
        endpoint.setdefault("provider_mode", "deterministic")
        endpoint.setdefault("model", "deterministic-bootstrap-v0.2")
    elif endpoint_type == "python_callable":
        endpoint.setdefault("provider_family", "custom")
        endpoint.setdefault("provider_mode", "custom")
        endpoint.setdefault("model", endpoint.get("callable_path", "python-callable"))
    elif endpoint_type == "openai_compatible":
        endpoint.setdefault("provider_family", "openai_compatible")
        endpoint.setdefault("provider_mode", "local")
        endpoint.setdefault("model", "unspecified-openai-compatible-model")
    return endpoint
