"""Manifest read/write helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import orjson

from hcaps.schema.manifest import DatasetManifest, FileRole, ManifestFile
from hcaps.store.base import PathLike, StoreWriteResult
from hcaps.utils.hashing import file_sha256
from hcaps.utils.time import utc_now


def write_manifest(path: PathLike, manifest: DatasetManifest) -> StoreWriteResult:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = orjson.dumps(manifest.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
    with output_path.open("wb") as handle:
        handle.write(payload)
        handle.write(b"\n")

    return StoreWriteResult(
        path=output_path,
        record_count=len(manifest.files),
        content_hash=file_sha256(output_path),
    )


def read_manifest(path: PathLike) -> DatasetManifest:
    input_path = Path(path)
    return DatasetManifest.model_validate_json(input_path.read_bytes())


def build_manifest_for_files(
    paths: Iterable[PathLike],
    *,
    dataset_name: str,
    dataset_version: str,
    schema_version: str = "0.1.0",
    generated_by: str = "hcaps.store.manifest",
    manifest_id: str | None = None,
) -> DatasetManifest:
    manifest_files: list[ManifestFile] = []
    capsule_count = 0

    for path_like in paths:
        path = Path(path_like)
        content_hash = file_sha256(path)
        role = _infer_file_role(path)
        record_count = _count_jsonl_records(path) if path.suffix == ".jsonl" else None
        if role == FileRole.CAPSULES_JSONL and record_count is not None:
            capsule_count += record_count

        manifest_files.append(
            ManifestFile(
                file_id=f"file_{content_hash[:16]}",
                path=str(path),
                role=role,
                media_type=_infer_media_type(path),
                size_bytes=path.stat().st_size,
                record_count=record_count,
                content_hash=content_hash,
            )
        )

    if manifest_id is None:
        joined_hashes = "".join(file.content_hash for file in manifest_files)
        manifest_id = f"manifest_{file_sha256_digest(joined_hashes)[:16]}"

    return DatasetManifest(
        manifest_id=manifest_id,
        schema_version=schema_version,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        created_at=utc_now(),
        generated_by=generated_by,
        capsule_count=capsule_count,
        files=manifest_files,
    )


def file_sha256_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _count_jsonl_records(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _infer_file_role(path: Path) -> FileRole:
    if path.suffix == ".jsonl":
        return FileRole.CAPSULES_JSONL
    if path.suffix == ".parquet":
        return FileRole.CAPSULES_PARQUET
    if path.suffix in {".yaml", ".yml", ".toml"}:
        return FileRole.CONFIG
    return FileRole.OTHER


def _infer_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "application/x-ndjson"
    if suffix == ".parquet":
        return "application/vnd.apache.parquet"
    if suffix in {".yaml", ".yml"}:
        return "application/yaml"
    if suffix == ".toml":
        return "application/toml"
    return "application/octet-stream"
