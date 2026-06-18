"""Provider ingress base contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hcaps.extraction.contracts import ExtractionTask, ProviderTrace
from hcaps.schema.capsule import UnitFloat
from hcaps.utils.hashing import canonical_json_bytes

FORBIDDEN_ACTIVE_KEYS = {
    "truth",
    "is_true",
    "correct",
    "is_correct",
    "factuality",
    "ground_truth",
    "label_truth",
    "proven_true",
}


class ProviderRequest(BaseModel):
    """Normalized provider request for one extraction task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: ExtractionTask
    input_text: str
    source_ref: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "0.1.0"
    template_version: str = "v0.1"
    context: dict[str, Any] = Field(default_factory=dict)

    @property
    def input_hash(self) -> str:
        return prefixed_sha256(
            {
                "task": self.task,
                "input_text": self.input_text,
                "source_ref": self.source_ref,
                "schema_version": self.schema_version,
                "template_version": self.template_version,
                "context": self.context,
            }
        )


class ProviderResponse(BaseModel):
    """Normalized response returned by any extraction provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_trace: ProviderTrace
    normalized_output: dict[str, Any] = Field(default_factory=dict)
    confidence: UnitFloat | None = None
    warnings: list[str] = Field(default_factory=list)
    raw_response: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_forbidden_active_keys(self) -> ProviderResponse:
        forbidden = find_forbidden_keys(self.normalized_output)
        if forbidden:
            joined = ", ".join(sorted(forbidden))
            msg = f"provider normalized output contains forbidden active keys: {joined}"
            raise ValueError(msg)
        return self


class ExtractionProvider(Protocol):
    """Provider protocol shared by deterministic, local, remote, and custom providers."""

    provider_id: str

    def run(self, request: ProviderRequest) -> ProviderResponse:
        """Run an extraction task and return normalized output."""


def prefixed_sha256(payload: Mapping[str, Any]) -> str:
    """Return a stable sha256:... hash for a JSON-compatible mapping."""

    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"sha256:{digest}"


def find_forbidden_keys(value: Any) -> set[str]:
    """Find forbidden active label keys in nested provider output."""

    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.casefold() in FORBIDDEN_ACTIVE_KEYS:
                found.add(key)
            found.update(find_forbidden_keys(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            found.update(find_forbidden_keys(nested))
    return found


def hash_normalized_output(output: Mapping[str, Any]) -> str:
    """Hash normalized provider output."""

    return prefixed_sha256(dict(output))
