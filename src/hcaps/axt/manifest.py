"""AXT bundle manifest models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import orjson
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_serializer

from hcaps.utils.time import utc_now

from .schema import AXT_COMPILER_VERSION, AXT_FORMAT_VERSION


class AxtArtifact(BaseModel):
    """One file written inside an AXT bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str
    role: str
    sha256: str
    byte_size: int = Field(ge=0)
    tensor_group: str | None = None
    tensors: list[str] = Field(default_factory=list)


class AxtManifest(BaseModel):
    """Top-level manifest for an AXT bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["AXT"] = "AXT"
    format_version: str = AXT_FORMAT_VERSION
    created_at: AwareDatetime
    compiler_version: str = AXT_COMPILER_VERSION
    input_path: str
    input_hash: str
    source_format: Literal["AXC", "AXP"]
    field_registry_hash: str
    vocabulary_registry_hash: str
    config_hash: str
    record_count: int = Field(ge=0)
    split_name: str | None = None
    tensor_groups: list[str]
    artifacts: list[AxtArtifact]
    hashes: dict[str, str]
    mask_summary: dict[str, int]
    negative_sample_summary: dict[str, int]
    provider_context_summary: dict[str, int]
    text_projection_summary: dict[str, Any]
    geometry_slot_summary: dict[str, int]
    warnings: list[str] = Field(default_factory=list)

    @field_serializer("created_at")
    def serialize_created_at(self, value: AwareDatetime) -> str:
        return value.isoformat()

    @classmethod
    def create(
        cls,
        *,
        input_path: str,
        input_hash: str,
        source_format: Literal["AXC", "AXP"],
        field_registry_hash: str,
        vocabulary_registry_hash: str,
        config_hash: str,
        record_count: int,
        split_name: str | None,
        tensor_groups: list[str],
        artifacts: list[AxtArtifact],
        hashes: dict[str, str],
        mask_summary: dict[str, int],
        negative_sample_summary: dict[str, int],
        provider_context_summary: dict[str, int],
        text_projection_summary: dict[str, Any],
        geometry_slot_summary: dict[str, int],
        warnings: list[str],
    ) -> AxtManifest:
        return cls(
            created_at=utc_now(),
            input_path=input_path,
            input_hash=input_hash,
            source_format=source_format,
            field_registry_hash=field_registry_hash,
            vocabulary_registry_hash=vocabulary_registry_hash,
            config_hash=config_hash,
            record_count=record_count,
            split_name=split_name,
            tensor_groups=tensor_groups,
            artifacts=artifacts,
            hashes=hashes,
            mask_summary=mask_summary,
            negative_sample_summary=negative_sample_summary,
            provider_context_summary=provider_context_summary,
            text_projection_summary=text_projection_summary,
            geometry_slot_summary=geometry_slot_summary,
            warnings=warnings,
        )


def write_manifest(path: str | Path, manifest: AxtManifest) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(
        orjson.dumps(
            manifest.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2
        )
        + b"\n"
    )


def load_manifest(path: str | Path) -> AxtManifest:
    return AxtManifest.model_validate_json(Path(path).read_bytes())
