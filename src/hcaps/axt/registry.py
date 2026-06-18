"""Field registry loading for AXT compilation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hcaps.format.hashing import file_hash

MIN_MARKDOWN_TABLE_SEPARATORS = 3


class FieldRegistryEntry(BaseModel):
    """One row from `spec/FIELD_REGISTRY_V1.md`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: str = Field(min_length=1)
    semantic_owner: str
    source_format: str
    axc_path: str
    axt_tensor_group: str
    axc_out_target_path: str
    visibility: str
    requiredness: str
    mask_behavior: str
    dtype: str
    vocabulary: str
    hash_behavior: str


class FieldRegistry(BaseModel):
    """Compiled view of the P2 field registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    entries: list[FieldRegistryEntry]

    def require_fields(self, field_names: set[str]) -> None:
        available = {entry.field_name for entry in self.entries}
        missing = sorted(field_names - available)
        if missing:
            msg = "field registry missing required fields: " + ", ".join(missing)
            raise ValueError(msg)

    def by_field_name(self, field_name: str) -> FieldRegistryEntry:
        for entry in self.entries:
            if entry.field_name == field_name:
                return entry
        msg = f"unknown field registry entry: {field_name}"
        raise KeyError(msg)

    def resolve_tensor_group(self, axc_path: str) -> str:
        for entry in self.entries:
            if entry.axc_path == axc_path:
                return entry.axt_tensor_group
        msg = f"AXC path is not registered: {axc_path}"
        raise KeyError(msg)

    def resolve_axc_out_target_path(self, axc_path: str) -> str:
        for entry in self.entries:
            if entry.axc_path == axc_path:
                return entry.axc_out_target_path
        msg = f"AXC path is not registered: {axc_path}"
        raise KeyError(msg)

    def resolve_visibility(self, field_name: str) -> str:
        return self.by_field_name(field_name).visibility

    def resolve_dtype(self, field_name: str) -> str:
        return self.by_field_name(field_name).dtype

    def resolve_mask_behavior(self, field_name: str) -> str:
        return self.by_field_name(field_name).mask_behavior

    def resolve_requiredness(self, field_name: str) -> str:
        return self.by_field_name(field_name).requiredness

    def to_json_dict(self) -> dict[str, object]:
        return {
            "source": self.path,
            "sha256": self.sha256,
            "entries": [entry.model_dump(mode="json") for entry in self.entries],
        }


REQUIRED_FIELD_NAMES: set[str] = {
    "capsule_id",
    "claim_family_id",
    "claim_state_id",
    "context_id",
    "canonical_text",
    "claim_type",
    "valid_as_of",
    "relation_type",
    "provenance_source",
    "negative_pool",
    "geometry_observable",
    "text_projection",
}


def load_field_registry(path: str | Path) -> FieldRegistry:
    registry_path = Path(path)
    rows = _parse_markdown_table(registry_path.read_text(encoding="utf-8"))
    entries = [FieldRegistryEntry.model_validate(row) for row in rows]
    registry = FieldRegistry(
        path=str(registry_path), sha256=file_hash(registry_path), entries=entries
    )
    registry.require_fields(REQUIRED_FIELD_NAMES)
    return registry


def _parse_markdown_table(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    headers: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or line.count("|") < MIN_MARKDOWN_TABLE_SEPARATORS:
            continue
        cells = [cell.strip().replace("`", "") for cell in line.strip("|").split("|")]
        if cells and cells[0] == "field_name":
            headers = [_normalize_header(cell) for cell in cells]
            continue
        if headers is None or cells[0].startswith("---"):
            continue
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def _normalize_header(value: str) -> str:
    replacements = {
        "AXC path": "axc_path",
        "AXT tensor group": "axt_tensor_group",
        "AXC-out target path": "axc_out_target_path",
        "mask behavior": "mask_behavior",
        "hash behavior": "hash_behavior",
        "allowed values / vocabulary": "vocabulary",
    }
    if value in replacements:
        return replacements[value]
    return value.replace(" ", "_").replace("-", "_")
