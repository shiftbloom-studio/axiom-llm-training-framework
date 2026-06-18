"""Provenance and sidecar metadata helpers for substrate builds."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class SidecarMetadata(BaseModel):
    """Strict optional metadata loaded from ``*.meta.json`` files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str | None = Field(default=None, min_length=1)
    authors: list[str] = Field(default_factory=list)
    publication_date: date | None = None
    license: str | None = Field(default=None, min_length=1)
    source_url: str | None = Field(default=None, min_length=1)
    domain: list[str] = Field(default_factory=list)
    temporal_cutoff: date | None = None

    @field_validator("authors", "domain")
    @classmethod
    def reject_empty_items(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            msg = "metadata lists may not contain empty strings"
            raise ValueError(msg)
        return value


def load_sidecar_metadata(path: Path) -> SidecarMetadata | None:
    """Load sidecar metadata if present, raising validation errors clearly."""

    sidecar_path = path.with_suffix(".meta.json")
    if not sidecar_path.exists():
        return None
    try:
        payload = orjson.loads(sidecar_path.read_bytes())
    except orjson.JSONDecodeError as exc:
        msg = f"invalid JSON sidecar metadata: {sidecar_path}"
        raise ValueError(msg) from exc
    try:
        return SidecarMetadata.model_validate(payload)
    except ValidationError as exc:
        msg = f"invalid sidecar metadata: {sidecar_path}: {exc}"
        raise ValueError(msg) from exc


def date_to_utc_datetime(value: date | datetime | None) -> datetime | None:
    """Convert a date-like value to a timezone-aware UTC datetime."""

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return datetime(value.year, value.month, value.day, tzinfo=UTC)
