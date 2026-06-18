"""Dataset manifest schema."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from hcaps.schema.document import SourceDocument
from hcaps.schema.identifiers import (
    SCHEMA_VERSION_PATTERN,
    SHA256_PATTERN,
    FileId,
    ManifestId,
)


class FileRole(StrEnum):
    """Role of a file tracked by a dataset manifest."""

    CAPSULES_JSONL = "capsules_jsonl"
    CAPSULES_PARQUET = "capsules_parquet"
    SOURCE_DOCUMENT = "source_document"
    CONFIG = "config"
    OTHER = "other"


class ManifestFile(BaseModel):
    """A content-addressed file entry in a dataset manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_id: FileId
    path: str = Field(min_length=1)
    role: FileRole
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    record_count: int | None = Field(default=None, ge=0)
    hash_algorithm: Literal["sha256"] = "sha256"
    content_hash: str = Field(pattern=SHA256_PATTERN)


class DatasetManifest(BaseModel):
    """Versioned manifest for reproducible Axiom claim-state datasets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_id: ManifestId
    schema_version: str = Field(pattern=SCHEMA_VERSION_PATTERN)
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    created_at: AwareDatetime
    generated_by: str = Field(min_length=1)
    capsule_count: int = Field(ge=0)
    files: list[ManifestFile] = Field(min_length=1)
    source_documents: list[SourceDocument] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
