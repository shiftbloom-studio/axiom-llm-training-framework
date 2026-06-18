"""JSON manifests for falsification harness runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArmManifest:
    """Manifest fragment for one generated arm."""

    arm_name: str
    artifact_path: str
    artifact_hash: str
    record_count: int
    transform_type: str
    render_mode: str | None
    included_fields: list[str]
    excluded_fields: list[str]
    audit_summary: dict[str, int]
    metrics_summary: dict[str, int | float]
    derived_noncanonical: bool = False
    capsule_artifact_path: str | None = None
    capsule_artifact_hash: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FalsificationRunManifest:
    """Top-level manifest for a deterministic falsification run."""

    run_id: str
    created_at: str
    axiom_version: str
    input_path: str
    input_hash: str
    source_format: str
    arms: list[ArmManifest]
    audits: dict[str, Any]
    metrics: dict[str, Any]
    deterministic_seed: int
    code_version_or_commit: str | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["arms"] = [arm.to_dict() for arm in self.arms]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp for manifests."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def stable_run_id(input_hash: str, arms: list[str], seed: int) -> str:
    """Create a stable run ID from input identity and run parameters."""

    payload = json.dumps(
        {"arms": arms, "input_hash": input_hash, "seed": seed},
        sort_keys=True,
    ).encode("utf-8")
    return f"falsification_{hashlib.sha256(payload).hexdigest()[:16]}"


def write_json(path: str | Path, payload: Any) -> None:
    """Write stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_run_manifest(path: str | Path) -> FalsificationRunManifest:
    """Load a falsification run manifest from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    arms = [ArmManifest(**arm) for arm in payload.pop("arms")]
    return FalsificationRunManifest(arms=arms, **payload)


def sha256_file(path: str | Path) -> str:
    """Hash one artifact file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash a file or a directory tree in stable path order."""

    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    digest = hashlib.sha256()
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()
