"""Content-addressed provider cache and replay helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_serializer

from hcaps.extraction.contracts import ProviderTrace
from hcaps.providers.base import (
    ExtractionProvider,
    ProviderRequest,
    ProviderResponse,
    prefixed_sha256,
)
from hcaps.providers.config import CacheMode, ProviderEndpointConfig
from hcaps.providers.errors import ProviderCacheMissError
from hcaps.utils.hashing import hash_record
from hcaps.utils.time import utc_now

CACHE_VERSION = "0.1.0"


class ProviderCacheRecord(BaseModel):
    """Serializable cache record for one provider request/response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_version: str = CACHE_VERSION
    request_hash: str = Field(min_length=1)
    provider: dict[str, Any]
    task: str = Field(min_length=1)
    created_at: AwareDatetime
    prompt_version: str = Field(min_length=1)
    input_hash: str = Field(min_length=1)
    output_hash: str = Field(min_length=1)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    normalized_output: dict[str, Any] = Field(default_factory=dict)
    provider_trace: ProviderTrace
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)


class ProviderCacheManifest(BaseModel):
    """Summary of cache records written under a cache root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root_path: Path
    record_count: int = Field(ge=0)
    provider_fingerprints: list[str] = Field(default_factory=list)
    request_hashes: list[str] = Field(default_factory=list)

    @field_serializer("root_path")
    def serialize_path(self, value: Path) -> str:
        return str(value)


class ProviderCache:
    """Content-addressed provider cache rooted at .cache/axiom/providers by default."""

    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)

    def provider_fingerprint(self, config: ProviderEndpointConfig) -> str:
        fingerprint_record = _provider_public_record(config)
        return hash_record(fingerprint_record)[:24]

    def request_hash(self, config: ProviderEndpointConfig, request: ProviderRequest) -> str:
        key_record = {
            "provider_type": config.type,
            "provider_base_url_fingerprint": _base_url_fingerprint(config.base_url),
            "provider_model": config.model,
            "prompt_template_version": request.template_version,
            "request_payload_hash": request.input_hash,
            "schema_version": request.schema_version,
            "task": request.task,
            "config_hash": hash_record(_provider_public_record(config)),
        }
        return prefixed_sha256(key_record)

    def path_for(self, config: ProviderEndpointConfig, request: ProviderRequest) -> Path:
        fingerprint = self.provider_fingerprint(config)
        request_hash = self.request_hash(config, request).removeprefix("sha256:")
        return self.root_path / fingerprint / f"{request_hash}.json"

    def read(self, config: ProviderEndpointConfig, request: ProviderRequest) -> ProviderResponse:
        path = self.path_for(config, request)
        if not path.exists():
            raise ProviderCacheMissError(f"provider cache miss: {path}")
        record = ProviderCacheRecord.model_validate_json(path.read_bytes())
        return ProviderResponse(
            provider_trace=record.provider_trace,
            normalized_output=record.normalized_output,
            confidence=record.confidence,
            warnings=record.warnings,
            raw_response=record.raw_response,
        )

    def write(
        self,
        config: ProviderEndpointConfig,
        request: ProviderRequest,
        response: ProviderResponse,
    ) -> Path:
        path = self.path_for(config, request)
        path.parent.mkdir(parents=True, exist_ok=True)
        request_hash = self.request_hash(config, request)
        trace = response.provider_trace.model_copy(update={"cache_key": request_hash})
        record = ProviderCacheRecord(
            request_hash=request_hash,
            provider=_provider_public_record(config),
            task=request.task,
            created_at=utc_now(),
            prompt_version=request.template_version,
            input_hash=request.input_hash,
            output_hash=trace.output_hash,
            raw_response=response.raw_response,
            normalized_output=response.normalized_output,
            provider_trace=trace,
            confidence=response.confidence,
            warnings=response.warnings,
        )
        path.write_bytes(
            orjson.dumps(
                record.model_dump(mode="json"),
                option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2,
            )
            + b"\n"
        )
        return path

    def replay_or_run(
        self,
        provider: ExtractionProvider,
        config: ProviderEndpointConfig,
        request: ProviderRequest,
        mode: CacheMode,
    ) -> ProviderResponse:
        if mode == "cache_only":
            return self.read(config, request)
        if mode == "live":
            try:
                return self.read(config, request)
            except ProviderCacheMissError:
                pass
        response = provider.run(request)
        self.write(config, request, response)
        cache_key = self.request_hash(config, request)
        return response.model_copy(
            update={
                "provider_trace": response.provider_trace.model_copy(
                    update={"cache_key": cache_key}
                )
            }
        )

    def manifest(self) -> ProviderCacheManifest:
        records = sorted(self.root_path.glob("*/*.json")) if self.root_path.exists() else []
        fingerprints = sorted({path.parent.name for path in records})
        return ProviderCacheManifest(
            root_path=self.root_path,
            record_count=len(records),
            provider_fingerprints=fingerprints,
            request_hashes=[f"sha256:{path.stem}" for path in records],
        )


def _provider_public_record(config: ProviderEndpointConfig) -> dict[str, Any]:
    return {
        "provider_id": config.provider_id,
        "type": config.type,
        "provider_family": config.provider_family,
        "provider_mode": config.provider_mode,
        "base_url_fingerprint": _base_url_fingerprint(config.base_url),
        "model": config.model,
        "api_key_env": config.api_key_env,
        "callable_path": config.callable_path,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "dry_run": config.dry_run,
    }


def _base_url_fingerprint(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    return prefixed_sha256({"base_url": base_url})
