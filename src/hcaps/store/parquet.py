"""Parquet storage backend.

This backend is intentionally conservative for Step 1. It writes a small set of
flat inspection columns plus the canonical nested capsule payload as
``record_json``. That makes the file usable with Arrow/Polars while preserving the
full contract. Native nested Parquet layouts are deferred until downstream
tensorization needs are known.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from hcaps.schema.capsule import HoloCapsule
from hcaps.schema.manifest import DatasetManifest
from hcaps.store.base import (
    CapsuleStoreError,
    InvalidCapsuleRecordError,
    PathLike,
    StoreWriteResult,
    ValidationReport,
)
from hcaps.store.manifest import read_manifest, write_manifest
from hcaps.utils.hashing import file_sha256


class ParquetCapsuleStore:
    """Parquet backend with canonical JSON preservation for nested fields."""

    limitations = (
        "Nested capsule fields are stored as canonical JSON in record_json. "
        "Only simple inspection columns are flattened in Step 1."
    )

    def write_capsules(
        self,
        path: PathLike,
        capsules: Iterable[HoloCapsule],
    ) -> StoreWriteResult:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, str]] = []
        for capsule in capsules:
            validated = HoloCapsule.model_validate(capsule)
            payload = validated.model_dump(mode="json")
            rows.append(
                {
                    "capsule_id": validated.capsule_id,
                    "schema_version": validated.schema_version,
                    "claim_id": validated.claim.claim_id,
                    "created_at": str(payload["created_at"]),
                    "updated_at": str(payload["updated_at"]),
                    "record_json": orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode(
                        "utf-8"
                    ),
                }
            )

        schema = pa.schema(
            [
                ("capsule_id", pa.string()),
                ("schema_version", pa.string()),
                ("claim_id", pa.string()),
                ("created_at", pa.string()),
                ("updated_at", pa.string()),
                ("record_json", pa.string()),
            ]
        )
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, output_path)  # type: ignore[no-untyped-call]
        return StoreWriteResult(
            path=output_path,
            record_count=len(rows),
            content_hash=file_sha256(output_path),
        )

    def read_capsules(self, path: PathLike) -> Iterator[HoloCapsule]:
        table = pq.read_table(Path(path))  # type: ignore[no-untyped-call]
        for row_number, row in enumerate(table.to_pylist(), start=1):
            record_json = row.get("record_json")
            if not isinstance(record_json, str):
                msg = f"{path}:{row_number}: missing record_json payload"
                raise CapsuleStoreError(msg)
            try:
                yield HoloCapsule.model_validate_json(record_json)
            except (ValidationError, ValueError) as exc:
                msg = f"{path}:{row_number}: invalid HoloCapsule parquet row: {exc}"
                raise InvalidCapsuleRecordError(msg) from exc

    def validate(self, path: PathLike) -> ValidationReport:
        input_path = Path(path)
        valid_count = 0
        errors: list[str] = []
        try:
            for _capsule in self.read_capsules(input_path):
                valid_count += 1
        except InvalidCapsuleRecordError as exc:
            errors.append(str(exc))
        return ValidationReport(path=input_path, valid_count=valid_count, errors=tuple(errors))

    def write_manifest(self, path: PathLike, manifest: DatasetManifest) -> StoreWriteResult:
        return write_manifest(path, manifest)

    def read_manifest(self, path: PathLike) -> DatasetManifest:
        return read_manifest(path)
