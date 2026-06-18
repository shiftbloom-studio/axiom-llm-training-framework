"""Storage interfaces and shared exceptions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hcaps.schema.capsule import HoloCapsule
from hcaps.schema.manifest import DatasetManifest

PathLike = str | Path


class CapsuleStoreError(Exception):
    """Base exception for storage failures."""


class InvalidCapsuleRecordError(CapsuleStoreError, ValueError):
    """Raised when a persisted record cannot validate as an Axiom claim-state capsule."""


@dataclass(frozen=True)
class StoreWriteResult:
    """Summary returned after writing a store artifact."""

    path: Path
    record_count: int
    content_hash: str


@dataclass(frozen=True)
class ValidationReport:
    """Validation summary for a persisted capsule file."""

    path: Path
    valid_count: int
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


class CapsuleStore(Protocol):
    """Protocol implemented by concrete Axiom claim-state storage backends."""

    def write_capsules(
        self,
        path: PathLike,
        capsules: Iterable[HoloCapsule],
    ) -> StoreWriteResult: ...

    def read_capsules(self, path: PathLike) -> Iterator[HoloCapsule]: ...

    def validate(self, path: PathLike) -> ValidationReport: ...

    def write_manifest(self, path: PathLike, manifest: DatasetManifest) -> StoreWriteResult: ...

    def read_manifest(self, path: PathLike) -> DatasetManifest: ...
