"""Typed configuration for AXC/AXP to AXT compilation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import orjson
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from hcaps.format.hashing import canonical_json_bytes, sha256_bytes

from .schema import TEXT_RENDER_MODES


class AxtCompileConfig(BaseModel):
    """Configuration for compiling structured Axiom records into AXT tensors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_path: Path
    output_path: Path
    field_registry_path: Path = Path("spec/FIELD_REGISTRY_V1.md")
    vocabulary_registry_path: Path = Path("spec/VOCABULARY_REGISTRY_V1.md")
    split_name: str | None = None
    allow_all_without_split: bool = False
    render_text_projection_modes: list[str] = Field(default_factory=lambda: list(TEXT_RENDER_MODES))
    max_text_length: int = Field(default=128, ge=1)
    include_provider_context: bool = True
    include_relation_neighborhoods: bool = True
    include_negative_samples: bool = True
    include_geometry_slots: bool = True
    include_evaluation_references: bool = False
    strict_temporal_masks: bool = True
    strict_schema_validation: bool = True
    hash_artifacts: bool = True
    seed: int = 13

    @field_serializer(
        "input_path",
        "output_path",
        "field_registry_path",
        "vocabulary_registry_path",
    )
    def serialize_path(self, value: Path) -> str:
        return str(value)

    def stable_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def stable_hash(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.stable_payload()))

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            orjson.dumps(
                self.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2
            )
            + b"\n"
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        config_path = Path(path)
        if not config_path.exists():
            msg = f"AXT compile config does not exist: {config_path}"
            raise FileNotFoundError(msg)
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        else:
            payload = orjson.loads(config_path.read_bytes())
        if not isinstance(payload, dict):
            msg = "AXT compile config must be a mapping"
            raise ValueError(msg)
        return cls.model_validate(_resolve_paths(payload, config_path.parent))


def load_axt_compile_config(path: str | Path) -> AxtCompileConfig:
    return AxtCompileConfig.from_file(path)


def _resolve_paths(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = dict(payload)
    for key in ("input_path", "output_path", "field_registry_path", "vocabulary_registry_path"):
        value = resolved.get(key)
        if isinstance(value, str):
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            resolved[key] = candidate
    return resolved
