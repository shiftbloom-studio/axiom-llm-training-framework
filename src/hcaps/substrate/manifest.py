"""Build configuration and manifest models for claim-field substrate runs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_serializer

from hcaps.schema.identifiers import SHA256_PATTERN


class BuildWarning(BaseModel):
    """A non-fatal substrate build warning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str | None = None
    severity: Literal["warning", "error"] = "warning"


class SubstrateBuildConfig(BaseModel):
    """Configuration for a deterministic source-to-capsule build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_path: Path
    output_path: Path
    manifest_path: Path
    axc_output_path: Path | None = None
    axp_package_path: Path | None = None
    schema_version: str = "0.1.0"
    dataset_name: str = "claim-field-substrate"
    pipeline_version: str = "0.2.0"
    max_chunk_chars: int = Field(default=1200, ge=200)
    chunk_overlap_chars: int = Field(default=0, ge=0)
    min_claim_chars: int = Field(default=32, ge=8)
    max_claim_chars: int = Field(default=480, ge=64)
    claim_family_similarity_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    cutoff_date: date | None = None
    strict_temporal_cutoff: bool = True
    invalid_sidecar_policy: Literal["warn", "error"] = "warn"

    @field_serializer(
        "input_path", "output_path", "manifest_path", "axc_output_path", "axp_package_path"
    )
    def serialize_path(self, value: Path | None) -> str | None:
        if value is None:
            return None
        return str(value)


class SubstrateBuildManifest(BaseModel):
    """Reproducibility manifest emitted beside built capsules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=r"^run_[a-f0-9]+$")
    package_version: str
    pipeline_version: str
    python_version: str
    timestamp: AwareDatetime
    input_paths: list[str]
    output_paths: list[str]
    config_hash: str = Field(pattern=SHA256_PATTERN)
    source_document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    candidate_claim_count: int = Field(ge=0)
    claim_family_count: int = Field(ge=0)
    relation_candidate_count: int = Field(ge=0)
    emitted_capsule_count: int = Field(ge=0)
    skipped_file_count: int = Field(ge=0)
    warnings: list[BuildWarning] = Field(default_factory=list)
    input_file_hashes: dict[str, str]
    output_file_hashes: dict[str, str]
    temporal_cutoff: str | None = None
