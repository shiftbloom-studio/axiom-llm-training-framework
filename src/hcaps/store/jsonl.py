"""JSONL storage backend with streaming validation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import orjson
from pydantic import ValidationError

from hcaps.schema.capsule import HoloCapsule
from hcaps.schema.manifest import DatasetManifest
from hcaps.store.base import InvalidCapsuleRecordError, PathLike, StoreWriteResult, ValidationReport
from hcaps.store.manifest import read_manifest, write_manifest
from hcaps.utils.hashing import file_sha256


class JsonlCapsuleStore:
    """Read and write newline-delimited Axiom claim-state records."""

    def write_capsules(
        self,
        path: PathLike,
        capsules: Iterable[HoloCapsule],
    ) -> StoreWriteResult:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        record_count = 0
        with output_path.open("wb") as handle:
            for capsule in capsules:
                validated = HoloCapsule.model_validate(capsule)
                payload = orjson.dumps(
                    validated.model_dump(mode="json"),
                    option=orjson.OPT_SORT_KEYS,
                )
                handle.write(payload)
                handle.write(b"\n")
                record_count += 1

        return StoreWriteResult(
            path=output_path,
            record_count=record_count,
            content_hash=file_sha256(output_path),
        )

    def read_capsules(self, path: PathLike) -> Iterator[HoloCapsule]:
        input_path = Path(path)
        with input_path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                yield _parse_jsonl_line(input_path, line_number, line)

    def validate(self, path: PathLike) -> ValidationReport:
        input_path = Path(path)
        valid_count = 0
        errors: list[str] = []
        with input_path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    _parse_jsonl_line(input_path, line_number, line)
                except InvalidCapsuleRecordError as exc:
                    errors.append(str(exc))
                else:
                    valid_count += 1

        return ValidationReport(path=input_path, valid_count=valid_count, errors=tuple(errors))

    def write_manifest(self, path: PathLike, manifest: DatasetManifest) -> StoreWriteResult:
        return write_manifest(path, manifest)

    def read_manifest(self, path: PathLike) -> DatasetManifest:
        return read_manifest(path)


def _parse_jsonl_line(path: Path, line_number: int, line: bytes) -> HoloCapsule:
    try:
        return HoloCapsule.model_validate_json(line)
    except (ValidationError, ValueError) as exc:
        msg = f"{path}:{line_number}: invalid Axiom claim-state record: {exc}"
        raise InvalidCapsuleRecordError(msg) from exc
