"""AXP package directory helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson
from pydantic import ValidationError

from hcaps.format.capsule import AxcCapsule
from hcaps.format.constants import AXC_EXTENSION, AXR_EXTENSION, AXSRC_EXTENSION
from hcaps.format.hashing import file_hash
from hcaps.format.identifiers import package_id
from hcaps.format.manifest import AxpConstruction, AxpCounts, AxpFileEntry, AxpManifest
from hcaps.format.streams import read_axc_stream, validate_axc_stream, write_axc_stream
from hcaps.format.validation import ValidationIssue, ValidationReport
from hcaps.utils.time import utc_now

PACKAGE_DIRS = (
    "data",
    "splits",
    "manifests",
    "compiled",
    "spec",
)
DEFAULT_SPLIT_FILES = (
    "train.json",
    "validation.json",
    "test.json",
    "temporal_holdout.json",
)
DEFAULT_MANIFEST_FILES = (
    "files.json",
    "hashes.json",
    "licenses.json",
    "construction_report.json",
    "leakage_report.json",
    "deduplication_report.json",
    "confound_controls.json",
)
REQUIRED_SPLIT_FILES = (
    "train.json",
    "validation.json",
    "temporal_holdout.json",
)
REQUIRED_MANIFEST_FILES = (
    "files.json",
    "hashes.json",
    "construction_report.json",
    "leakage_report.json",
)
CAPSULE_STREAM_PATH = f"data/capsules{AXC_EXTENSION}"
SOURCE_STREAM_PATH = f"data/sources{AXSRC_EXTENSION}"
RELATION_STREAM_PATH = f"data/relations{AXR_EXTENSION}"
LEGACY_CAPSULE_EXTENSION = ".axc.jsonl"
LEGACY_SOURCE_STREAM_PATH = "data/sources.jsonl"
LEGACY_RELATION_STREAM_PATH = "data/relations.jsonl"


def create_package_skeleton(
    path: str | Path, *, dataset_name: str, license: str = "unknown"
) -> AxpManifest:
    package_path = Path(path)
    package_path.mkdir(parents=True, exist_ok=True)
    for directory in PACKAGE_DIRS:
        (package_path / directory).mkdir(exist_ok=True)
    (package_path / "README.md").write_text(
        f"# {dataset_name}\n\nAXP package for AXF claim-state data.\n",
        encoding="utf-8",
    )
    (package_path / "compiled" / "README.md").write_text(
        "# AXT Tensor Bundle\n\nAXT is specified in AXF v0.1, but tensor compilation "
        "belongs to the later training bridge step.\n",
        encoding="utf-8",
    )
    for split_file in DEFAULT_SPLIT_FILES:
        _write_json(package_path / "splits" / split_file, {"capsule_ids": []})
    for manifest_file in DEFAULT_MANIFEST_FILES:
        _write_json(package_path / "manifests" / manifest_file, {})
    _write_json(package_path / "spec" / "axf_version.json", {"axf": "0.1.0"})
    _write_json(package_path / "spec" / "schema_snapshot.json", {"schema": "AXF v0.1"})
    manifest = AxpManifest(
        package_id=package_id(dataset_name, "0.1.0"),
        dataset_name=dataset_name,
        created_at=utc_now(),
        license=license,
        construction=AxpConstruction(
            builder="axiom-package",
            builder_version="0.1.0",
        ),
    )
    write_manifest(package_path, manifest)
    return manifest


def write_manifest(package_path: str | Path, manifest: AxpManifest) -> None:
    path = Path(package_path) / "axiom.json"
    payload = orjson.dumps(
        manifest.model_dump(mode="json", by_alias=True),
        option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2,
    )
    path.write_bytes(payload + b"\n")


def load_manifest(package_path: str | Path) -> AxpManifest:
    return AxpManifest.model_validate_json((Path(package_path) / "axiom.json").read_bytes())


def list_capsule_streams(package_path: str | Path) -> list[Path]:
    data_dir = Path(package_path) / "data"
    streams = {*data_dir.glob(f"*{AXC_EXTENSION}"), *data_dir.glob(f"*{LEGACY_CAPSULE_EXTENSION}")}
    return sorted(streams)


def validate_package(path: str | Path) -> ValidationReport:
    package_path = Path(path)
    if not package_path.exists():
        return ValidationReport(
            ok=False,
            issues=[
                ValidationIssue(code="missing_package", message=f"missing package: {package_path}")
            ],
        )
    issues = _required_layout_issues(package_path)
    if issues:
        return ValidationReport(ok=False, issues=issues)

    manifest = _load_manifest_or_report(package_path)
    if isinstance(manifest, ValidationReport):
        return manifest
    _validate_manifest_file_entries(package_path, manifest, issues)
    _validate_hash_manifest(package_path, issues)
    capsule_ids, claim_family_ids, source_ids = _validate_capsule_streams(package_path, issues)
    _validate_split_references(package_path, capsule_ids, issues)
    _validate_source_metadata(package_path, source_ids, issues)
    _validate_relation_metadata(package_path, capsule_ids, claim_family_ids, issues)
    return ValidationReport(ok=not issues, issues=issues)


def build_package_from_axc_stream(
    package_path: str | Path,
    *,
    dataset_name: str,
    capsules: Iterable[AxcCapsule],
    license: str = "unknown",
    builder: str = "claim-field-substrate-builder",
    builder_version: str = "0.2.0",
) -> AxpManifest:
    package = Path(package_path)
    create_package_skeleton(package, dataset_name=dataset_name, license=license)
    capsule_records = list(capsules)
    capsule_path = package / CAPSULE_STREAM_PATH
    record_count = write_axc_stream(capsule_path, capsule_records)
    source_count = _write_source_stream(package / SOURCE_STREAM_PATH, capsule_records)
    relation_count = _write_relation_stream(package / RELATION_STREAM_PATH, capsule_records)
    manifest_entry_paths = [
        (CAPSULE_STREAM_PATH, "capsule_stream", "application/x-ndjson; profile=AXC", record_count),
        (
            SOURCE_STREAM_PATH,
            "source_metadata",
            "application/x-ndjson; profile=AXSRC",
            source_count,
        ),
        (
            RELATION_STREAM_PATH,
            "relation_metadata",
            "application/x-ndjson; profile=AXR",
            relation_count,
        ),
        ("splits/train.json", "split", "application/json", 0),
        ("splits/validation.json", "split", "application/json", 0),
        ("splits/temporal_holdout.json", "split", "application/json", 0),
        ("manifests/construction_report.json", "construction_report", "application/json", None),
        ("manifests/leakage_report.json", "leakage_report", "application/json", None),
    ]
    file_entries = [
        _file_entry(package, relative_path, role, media_type, entry_count)
        for relative_path, role, media_type, entry_count in manifest_entry_paths
    ]
    hashes = {entry.relative_path: entry.sha256 for entry in file_entries}
    _write_json(
        package / "manifests" / "files.json",
        [entry.model_dump(mode="json") for entry in file_entries],
    )
    _write_json(package / "manifests" / "hashes.json", hashes)
    package_hash_seed = hashes[CAPSULE_STREAM_PATH][:16]
    manifest = AxpManifest(
        package_id=package_id(dataset_name, package_hash_seed),
        dataset_name=dataset_name,
        created_at=utc_now(),
        license=license,
        files=file_entries,
        counts=AxpCounts(
            capsules=record_count,
            sources=source_count,
            relations=relation_count,
        ),
        construction=AxpConstruction(
            builder=builder,
            builder_version=builder_version,
        ),
        hashes=hashes,
    )
    write_manifest(package, manifest)
    return manifest


def inspect_package(path: str | Path) -> dict[str, object]:
    manifest = load_manifest(path)
    streams = list_capsule_streams(path)
    return {
        "package_id": manifest.package_id,
        "dataset_name": manifest.dataset_name,
        "capsule_streams": [str(stream) for stream in streams],
        "capsule_count": sum(1 for stream in streams for _capsule in read_axc_stream(stream)),
        "file_count": len(manifest.files),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n"
    )


def _write_source_stream(path: Path, capsules: list[AxcCapsule]) -> int:
    records: dict[str, dict[str, object]] = {}
    for capsule in capsules:
        for source in capsule.provenance.sources:
            records[source.source_id] = source.model_dump(mode="json")
    _write_jsonl(path, (records[source_id] for source_id in sorted(records)))
    return len(records)


def _write_relation_stream(path: Path, capsules: list[AxcCapsule]) -> int:
    family_to_capsule = {
        capsule.ids.claim_family_id: capsule.ids.capsule_id for capsule in capsules
    }
    records: list[dict[str, object]] = []
    for capsule in capsules:
        for relation in capsule.relations:
            target_capsule_id = family_to_capsule.get(relation.target_claim_family_id)
            if target_capsule_id is None:
                continue
            records.append(
                {
                    "relation_id": relation.relation_id,
                    "source_capsule_id": capsule.ids.capsule_id,
                    "target_capsule_id": target_capsule_id,
                    "source_claim_family_id": capsule.ids.claim_family_id,
                    "target_claim_family_id": relation.target_claim_family_id,
                    "relation_type": relation.relation_type,
                    "confidence": relation.confidence,
                    "extraction_method": relation.extraction_method,
                }
            )
    records.sort(key=lambda record: str(record["relation_id"]))
    _write_jsonl(path, records)
    return len(records)


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record, option=orjson.OPT_SORT_KEYS))
            handle.write(b"\n")


def _file_entry(
    package: Path,
    relative_path: str,
    role: str,
    media_type: str,
    record_count: int | None,
) -> AxpFileEntry:
    file_path = package / relative_path
    return AxpFileEntry(
        relative_path=relative_path,
        role=role,
        media_type=media_type,
        byte_size=file_path.stat().st_size,
        sha256=file_hash(file_path),
        record_count=record_count,
    )


def _required_layout_issues(package_path: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not (package_path / "axiom.json").exists():
        issues.append(ValidationIssue(code="missing_manifest", message="missing axiom.json"))
    if not (package_path / "README.md").exists():
        issues.append(ValidationIssue(code="missing_required_file", message="missing README.md"))
    issues.extend(_missing_directory_issues(package_path))
    issues.extend(_missing_split_issues(package_path))
    issues.extend(_missing_manifest_file_issues(package_path))
    return issues


def _missing_directory_issues(package_path: Path) -> list[ValidationIssue]:
    return [
        ValidationIssue(code="missing_directory", message=f"missing {directory}/")
        for directory in PACKAGE_DIRS
        if not (package_path / directory).is_dir()
    ]


def _missing_split_issues(package_path: Path) -> list[ValidationIssue]:
    return [
        ValidationIssue(
            code="missing_required_file",
            message=f"missing splits/{split_file}",
            path=f"splits/{split_file}",
        )
        for split_file in REQUIRED_SPLIT_FILES
        if not (package_path / "splits" / split_file).exists()
    ]


def _missing_manifest_file_issues(package_path: Path) -> list[ValidationIssue]:
    return [
        ValidationIssue(
            code="missing_required_file",
            message=f"missing manifests/{manifest_file}",
            path=f"manifests/{manifest_file}",
        )
        for manifest_file in REQUIRED_MANIFEST_FILES
        if not (package_path / "manifests" / manifest_file).exists()
    ]


def _load_manifest_or_report(package_path: Path) -> AxpManifest | ValidationReport:
    try:
        return load_manifest(package_path)
    except ValidationError as exc:
        code = "axp_manifest_validation_failed"
        raw_manifest = _load_json(package_path / "axiom.json")
        if isinstance(raw_manifest, dict) and raw_manifest.get("format_version") != "0.1.0":
            code = "unsupported_format_version"
        return ValidationReport(
            ok=False,
            issues=[
                ValidationIssue(
                    code=code,
                    message=str(exc),
                    path="axiom.json",
                )
            ],
        )


def _validate_manifest_file_entries(
    package_path: Path,
    manifest: AxpManifest,
    issues: list[ValidationIssue],
) -> None:
    for file_entry in manifest.files:
        file_path = package_path / file_entry.relative_path
        if not file_path.exists():
            issues.append(
                ValidationIssue(
                    code="missing_manifest_file",
                    message=f"manifest file is missing: {file_entry.relative_path}",
                    path=file_entry.relative_path,
                )
            )
            continue
        if file_hash(file_path) != file_entry.sha256:
            issues.append(
                ValidationIssue(
                    code="file_hash_mismatch",
                    message=f"hash mismatch for {file_entry.relative_path}",
                    path=file_entry.relative_path,
                )
            )


def _load_json(path: Path) -> Any:
    try:
        return orjson.loads(path.read_bytes())
    except Exception:
        return None


def _validate_hash_manifest(package_path: Path, issues: list[ValidationIssue]) -> None:
    hashes_path = package_path / "manifests" / "hashes.json"
    hashes = _load_json(hashes_path)
    if hashes in (None, {}):
        return
    if not isinstance(hashes, dict):
        issues.append(
            ValidationIssue(
                code="invalid_hash_manifest",
                message="manifests/hashes.json must contain an object",
                path="manifests/hashes.json",
            )
        )
        return
    for relative_path, expected_hash in hashes.items():
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            issues.append(
                ValidationIssue(
                    code="invalid_hash_manifest",
                    message="hash manifest entries must map string paths to string hashes",
                    path="manifests/hashes.json",
                )
            )
            continue
        file_path = package_path / relative_path
        if not file_path.exists():
            issues.append(
                ValidationIssue(
                    code="missing_hash_manifest_file",
                    message=f"hash manifest file is missing: {relative_path}",
                    path=relative_path,
                )
            )
            continue
        if file_hash(file_path) != expected_hash:
            issues.append(
                ValidationIssue(
                    code="hash_manifest_mismatch",
                    message=f"hash manifest mismatch for {relative_path}",
                    path=relative_path,
                )
            )


def _validate_capsule_streams(
    package_path: Path,
    issues: list[ValidationIssue],
) -> tuple[set[str], set[str], set[str]]:
    capsule_ids: set[str] = set()
    claim_family_ids: set[str] = set()
    source_ids: set[str] = set()
    for stream in list_capsule_streams(package_path):
        relative_path = stream.relative_to(package_path).as_posix()
        stream_report = validate_axc_stream(stream)
        for issue in stream_report.issues:
            issues.append(issue.model_copy(update={"path": relative_path}))
        try:
            capsules = list(read_axc_stream(stream))
        except ValueError as exc:
            issues.append(
                ValidationIssue(
                    code="invalid_capsule_stream",
                    message=str(exc),
                    path=relative_path,
                )
            )
            continue
        for capsule in capsules:
            capsule_ids.add(capsule.ids.capsule_id)
            claim_family_ids.add(capsule.ids.claim_family_id)
            source_ids.update(source.source_id for source in capsule.provenance.sources)
    return capsule_ids, claim_family_ids, source_ids


def _validate_split_references(
    package_path: Path,
    capsule_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    for split_file in REQUIRED_SPLIT_FILES:
        relative_path = f"splits/{split_file}"
        split_payload = _load_json(package_path / relative_path)
        if split_payload is None:
            issues.append(
                ValidationIssue(
                    code="invalid_split_file",
                    message=f"{relative_path} must contain JSON",
                    path=relative_path,
                )
            )
            continue
        if isinstance(split_payload, dict):
            referenced_ids = split_payload.get("capsule_ids", [])
        else:
            referenced_ids = split_payload
        if not isinstance(referenced_ids, list):
            issues.append(
                ValidationIssue(
                    code="invalid_split_file",
                    message=f"{relative_path} must contain a capsule_ids list",
                    path=relative_path,
                )
            )
            continue
        for referenced_id in referenced_ids:
            if not isinstance(referenced_id, str) or referenced_id not in capsule_ids:
                issues.append(
                    ValidationIssue(
                        code="missing_capsule_reference",
                        message=f"{relative_path} references missing capsule_id {referenced_id!r}",
                        path=relative_path,
                    )
                )


def _validate_source_metadata(
    package_path: Path,
    capsule_source_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    source_path = _first_existing(
        package_path,
        SOURCE_STREAM_PATH,
        LEGACY_SOURCE_STREAM_PATH,
    )
    if not source_path.exists():
        return
    relative_path = source_path.relative_to(package_path).as_posix()
    metadata_source_ids: set[str] = set()
    for line_number, line in enumerate(source_path.read_bytes().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    code="invalid_source_metadata",
                    message=str(exc),
                    path=relative_path,
                    line_number=line_number,
                )
            )
            continue
        source_id_value = record.get("source_id") if isinstance(record, dict) else None
        if isinstance(source_id_value, str):
            metadata_source_ids.add(source_id_value)
    for source_id_value in sorted(capsule_source_ids - metadata_source_ids):
        issues.append(
            ValidationIssue(
                code="missing_source_metadata",
                message=f"capsule source_id missing from {relative_path}: {source_id_value}",
                path=relative_path,
            )
        )


def _validate_relation_metadata(
    package_path: Path,
    capsule_ids: set[str],
    claim_family_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    relation_path = _first_existing(
        package_path,
        RELATION_STREAM_PATH,
        LEGACY_RELATION_STREAM_PATH,
    )
    if not relation_path.exists():
        return
    relative_path = relation_path.relative_to(package_path).as_posix()
    for line_number, line in enumerate(relation_path.read_bytes().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    code="invalid_relation_metadata",
                    message=str(exc),
                    path=relative_path,
                    line_number=line_number,
                )
            )
            continue
        if not isinstance(record, dict):
            issues.append(
                ValidationIssue(
                    code="invalid_relation_metadata",
                    message="relation records must be JSON objects",
                    path=relative_path,
                    line_number=line_number,
                )
            )
            continue
        source_family = record.get("source_claim_family_id")
        target_family = record.get("target_claim_family_id")
        source_capsule = record.get("source_capsule_id")
        target_capsule = record.get("target_capsule_id")
        if source_family is not None and source_family not in claim_family_ids:
            issues.append(
                ValidationIssue(
                    code="missing_relation_source",
                    message=f"relation source claim family is missing: {source_family}",
                    path=relative_path,
                    line_number=line_number,
                )
            )
        if target_family is not None and target_family not in claim_family_ids:
            issues.append(
                ValidationIssue(
                    code="missing_relation_target",
                    message=f"relation target claim family is missing: {target_family}",
                    path=relative_path,
                    line_number=line_number,
                )
            )
        if source_capsule is not None and source_capsule not in capsule_ids:
            issues.append(
                ValidationIssue(
                    code="missing_relation_source",
                    message=f"relation source capsule is missing: {source_capsule}",
                    path=relative_path,
                    line_number=line_number,
                )
            )
        if target_capsule is not None and target_capsule not in capsule_ids:
            issues.append(
                ValidationIssue(
                    code="missing_relation_target",
                    message=f"relation target capsule is missing: {target_capsule}",
                    path=relative_path,
                    line_number=line_number,
                )
            )


def _first_existing(package_path: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        candidate = package_path / relative_path
        if candidate.exists():
            return candidate
    return package_path / relative_paths[0]
