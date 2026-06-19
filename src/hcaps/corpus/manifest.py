"""Corpus build configuration and manifest models."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_serializer

from hcaps.providers.config import ProviderIngressConfig


class CorpusBuildConfig(BaseModel):
    """Configuration for a provider-aware corpus build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_id: str = Field(min_length=1)
    corpus_version: str = "0.1.0"
    domain: list[str] = Field(default_factory=list)
    input_path: Path
    output_dir: Path
    dataset_name: str = "ml_software_benchmarks"
    cutoff_date: date | None = None
    max_chunk_chars: int = Field(default=1200, ge=200)
    claim_family_similarity_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    provider_config_path: Path | None = None
    providers: ProviderIngressConfig | None = None
    strict_temporal_cutoff: bool = True
    pdf_reader_enabled: bool = False
    pdf_reader_required: bool = False

    @field_serializer("input_path", "output_dir", "provider_config_path")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None

    def to_yaml(self, path: str | Path) -> None:
        """Write this config to a YAML file for reproducibility."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )


class CorpusManifest(BaseModel):
    """Top-level reproducibility manifest for one corpus build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_id: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1)
    created_at: AwareDatetime
    domain: list[str] = Field(default_factory=list)
    source_count: int = Field(ge=0)
    claim_family_count: int = Field(ge=0)
    capsule_count: int = Field(ge=0)
    relation_candidate_count: int = Field(ge=0)
    provider_policy: dict[str, Any]
    license_summary: dict[str, Any]
    temporal_policy: dict[str, Any]
    hashes: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class CorpusBuildResult(BaseModel):
    """In-memory summary returned by the corpus builder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: CorpusManifest
    artifact_paths: dict[str, Path]
    package_path: Path

    @field_serializer("artifact_paths")
    def serialize_artifact_paths(self, value: dict[str, Path]) -> dict[str, str]:
        return {key: str(path) for key, path in value.items()}

    @field_serializer("package_path")
    def serialize_package_path(self, value: Path) -> str:
        return str(value)


def load_corpus_build_config(path: str | Path) -> CorpusBuildConfig:
    """Load a corpus build config from YAML."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        msg = "corpus config must be a mapping"
        raise ValueError(msg)
    resolved = _resolve_paths(payload, config_path.parent)
    return CorpusBuildConfig.model_validate(resolved)


def _resolve_paths(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = dict(payload)
    for key in ("input_path", "output_dir", "provider_config_path"):
        value = resolved.get(key)
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            resolved[key] = path
    return resolved
