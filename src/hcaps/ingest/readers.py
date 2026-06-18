"""Local source readers for the claim-field substrate builder."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
from pydantic import ValidationError

from hcaps.ingest.documents import SourceDocument
from hcaps.provenance import SidecarMetadata, date_to_utc_datetime, load_sidecar_metadata
from hcaps.substrate.canonicalize import normalize_text, stable_id
from hcaps.substrate.manifest import BuildWarning, SubstrateBuildConfig
from hcaps.utils.hashing import file_sha256

_PdfReader: Any
try:
    from pypdf import PdfReader as _ImportedPdfReader
except ImportError:  # pragma: no cover - optional dependency.
    _PdfReader = None
else:
    _PdfReader = _ImportedPdfReader

READER_VERSION = "0.2.0"
SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | {".jsonl", ".pdf"}


class ReaderError(ValueError):
    """Raised when source ingestion cannot continue."""


def read_source_documents(
    input_path: Path,
    config: SubstrateBuildConfig,
) -> tuple[list[SourceDocument], list[BuildWarning], int]:
    """Read supported documents from a file or directory in deterministic order."""

    if not input_path.exists():
        msg = f"input path does not exist: {input_path}"
        raise FileNotFoundError(msg)

    warnings: list[BuildWarning] = []
    skipped = 0
    documents: list[SourceDocument] = []
    for path in _iter_source_files(input_path):
        if path.name.endswith(".meta.json"):
            continue
        if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            skipped += 1
            warnings.append(
                BuildWarning(
                    code="unsupported_file_type",
                    message=f"skipped unsupported source file type: {path.suffix}",
                    path=str(path),
                )
            )
            continue
        try:
            sidecar = load_sidecar_metadata(path)
        except (ValidationError, ValueError) as exc:
            if config.invalid_sidecar_policy == "error":
                raise ReaderError(str(exc)) from exc
            sidecar = None
            warnings.append(BuildWarning(code="invalid_sidecar", message=str(exc), path=str(path)))

        if path.suffix.casefold() == ".jsonl":
            read_documents, read_warnings = _read_jsonl_records(path, sidecar)
            documents.extend(read_documents)
            warnings.extend(read_warnings)
        elif path.suffix.casefold() == ".pdf":
            read_documents, read_warnings, read_skipped = _read_pdf_documents(
                path,
                sidecar,
                config,
            )
            documents.extend(read_documents)
            warnings.extend(read_warnings)
            skipped += read_skipped
        else:
            documents.append(_read_plain_text_document(path, sidecar))

    return documents, warnings, skipped


def input_file_hashes(input_path: Path) -> dict[str, str]:
    """Return hashes for all regular input files, including sidecars."""

    if input_path.is_file():
        return {str(input_path): file_sha256(input_path)}
    return {
        str(path): file_sha256(path) for path in sorted(input_path.rglob("*")) if path.is_file()
    }


def _iter_source_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    yield from sorted(path for path in input_path.rglob("*") if path.is_file())


def _read_plain_text_document(path: Path, metadata: SidecarMetadata | None) -> SourceDocument:
    raw_bytes = path.read_bytes()
    text = normalize_text(raw_bytes.decode("utf-8"))
    return _build_source_document(
        path=path,
        record_key=str(path),
        raw_bytes=raw_bytes,
        text=text,
        metadata=metadata,
        title=metadata.title if metadata and metadata.title else path.stem.replace("_", " "),
        reader_name=f"plain-text:{path.suffix.casefold().lstrip('.')}",
        page_number=None,
    )


def _read_jsonl_records(
    path: Path,
    metadata: SidecarMetadata | None,
) -> tuple[list[SourceDocument], list[BuildWarning]]:
    documents: list[SourceDocument] = []
    warnings: list[BuildWarning] = []
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            warnings.append(
                BuildWarning(
                    code="invalid_jsonl_source_record",
                    message=f"invalid JSONL record at line {line_number}: {exc}",
                    path=str(path),
                )
            )
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            warnings.append(
                BuildWarning(
                    code="invalid_jsonl_source_record",
                    message=f"JSONL record at line {line_number} must contain a text field",
                    path=str(path),
                )
            )
            continue
        record_metadata = _metadata_from_record(payload, metadata)
        text = normalize_text(payload["text"])
        documents.append(
            _build_source_document(
                path=path,
                record_key=f"{path}#L{line_number}",
                raw_bytes=line,
                text=text,
                metadata=record_metadata,
                title=str(
                    payload.get("title") or record_metadata.title or f"{path.stem} {line_number}"
                ),
                reader_name="jsonl:text-record",
                page_number=None,
            )
        )
    return documents, warnings


def _read_pdf_documents(
    path: Path,
    metadata: SidecarMetadata | None,
    config: SubstrateBuildConfig,
) -> tuple[list[SourceDocument], list[BuildWarning], int]:
    if not config.pdf_reader_enabled:
        message = (
            "PDF reader interface is disabled; enable pdf_reader_enabled with an optional "
            "pypdf installation or provide text/markdown sources"
        )
        if config.pdf_reader_required:
            raise ReaderError(message)
        return (
            [],
            [BuildWarning(code="pdf_reader_disabled", message=message, path=str(path))],
            1,
        )
    if _PdfReader is None:
        message = "PDF reader requires optional dependency pypdf"
        if config.pdf_reader_required:
            raise ReaderError(message)
        return (
            [],
            [BuildWarning(code="pdf_reader_unavailable", message=message, path=str(path))],
            1,
        )
    try:
        reader = _PdfReader(str(path))
    except Exception as exc:
        message = f"PDF parser failed: {exc}"
        if config.pdf_reader_required:
            raise ReaderError(message) from exc
        return (
            [],
            [BuildWarning(code="pdf_reader_failed", message=message, path=str(path))],
            1,
        )

    raw_bytes = path.read_bytes()
    documents: list[SourceDocument] = []
    warnings: list[BuildWarning] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = normalize_text(page.extract_text() or "")
        except Exception as exc:
            warnings.append(
                BuildWarning(
                    code="pdf_page_extract_failed",
                    message=f"PDF page {page_index} extraction failed: {exc}",
                    path=str(path),
                )
            )
            continue
        if not text:
            warnings.append(
                BuildWarning(
                    code="pdf_page_empty",
                    message=f"PDF page {page_index} produced no extractable text",
                    path=str(path),
                )
            )
            continue
        title = metadata.title if metadata and metadata.title else path.stem.replace("_", " ")
        documents.append(
            _build_source_document(
                path=path,
                record_key=f"{path}#page={page_index}",
                raw_bytes=raw_bytes + f"#page={page_index}".encode(),
                text=text,
                metadata=metadata,
                title=f"{title} p. {page_index}",
                reader_name="pdf:pypdf",
                page_number=page_index,
            )
        )
    if not documents:
        warnings.append(
            BuildWarning(
                code="pdf_no_extractable_text",
                message="PDF produced no extractable text pages",
                path=str(path),
            )
        )
        return [], warnings, 1
    return documents, warnings, 0


def _metadata_from_record(
    payload: dict[str, Any],
    fallback: SidecarMetadata | None,
) -> SidecarMetadata:
    metadata_payload: dict[str, Any] = fallback.model_dump(mode="json") if fallback else {}
    for key in ("title", "authors", "publication_date", "license", "source_url", "domain"):
        if key in payload:
            metadata_payload[key] = payload[key]
    return SidecarMetadata.model_validate(metadata_payload)


def _build_source_document(
    *,
    path: Path,
    record_key: str,
    raw_bytes: bytes,
    text: str,
    metadata: SidecarMetadata | None,
    title: str,
    reader_name: str,
    page_number: int | None,
) -> SourceDocument:
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    normalized_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    document_id = stable_id("doc", record_key, normalized_hash)
    source_id = stable_id("src", record_key, raw_hash)
    published_at = date_to_utc_datetime(metadata.publication_date) if metadata else None
    temporal_cutoff = date_to_utc_datetime(metadata.temporal_cutoff) if metadata else None
    return SourceDocument(
        document_id=document_id,
        source_id=source_id,
        source_path=str(path.resolve()),
        file_name=path.name,
        media_type=_media_type(path),
        raw_sha256=raw_hash,
        normalized_text_sha256=normalized_hash,
        text=text,
        text_length=len(text),
        title=title,
        authors=metadata.authors if metadata else [],
        published_at=published_at,
        license=metadata.license if metadata and metadata.license else "unknown",
        source_url=metadata.source_url if metadata else None,
        domains=metadata.domain if metadata else [],
        temporal_cutoff_at=temporal_cutoff,
        page_number=page_number,
        reader_name=reader_name,
        reader_version=READER_VERSION,
    )


def _media_type(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".txt":
        return "text/plain"
    if suffix in {".md", ".markdown"}:
        return "text/markdown"
    if suffix == ".jsonl":
        return "application/x-ndjson"
    if suffix == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


UNKNOWN_SOURCE_TIME = datetime(1970, 1, 1, tzinfo=UTC)
