"""Document models used by the claim-field substrate builder."""

from __future__ import annotations

from datetime import datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from hcaps.schema.identifiers import ChunkId, DocumentId, SourceId


class SourceDocument(BaseModel):
    """Normalized source document with provenance needed for capsule construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: DocumentId
    source_id: SourceId
    source_path: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    raw_sha256: str = Field(min_length=64, max_length=64)
    normalized_text_sha256: str = Field(min_length=64, max_length=64)
    text: str = Field(min_length=1)
    text_length: int = Field(ge=1)
    title: str
    authors: list[str] = Field(default_factory=list)
    published_at: AwareDatetime | None = None
    license: str = "unknown"
    source_url: str | None = None
    domains: list[str] = Field(default_factory=list)
    temporal_cutoff_at: AwareDatetime | None = None
    page_number: int | None = Field(default=None, ge=1)
    reader_name: str = Field(min_length=1)
    reader_version: str = Field(min_length=1)


class DocumentChunk(BaseModel):
    """Stable source chunk preserving offsets and section context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: ChunkId
    document_id: DocumentId
    source_id: SourceId
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    section_path: list[str] = Field(default_factory=list)
    page_number: int | None = Field(default=None, ge=1)
    source_timestamp: datetime | None = None
