from __future__ import annotations

from pathlib import Path

import pytest

from hcaps.ingest.readers import ReaderError, read_source_documents
from hcaps.substrate.manifest import SubstrateBuildConfig

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "source_docs"


def _config(input_path: Path, tmp_path: Path, *, policy: str = "warn") -> SubstrateBuildConfig:
    return SubstrateBuildConfig(
        input_path=input_path,
        output_path=tmp_path / "capsules.jsonl",
        manifest_path=tmp_path / "build_manifest.json",
        invalid_sidecar_policy=policy,  # type: ignore[arg-type]
    )


def test_reads_txt_and_markdown_with_sidecar_metadata(tmp_path: Path) -> None:
    documents, warnings, skipped = read_source_documents(
        FIXTURE_DOCS, _config(FIXTURE_DOCS, tmp_path)
    )

    media_types = {document.media_type for document in documents}
    titles = {document.title for document in documents}

    assert skipped == 0
    assert warnings == []
    assert "text/plain" in media_types
    assert "text/markdown" in media_types
    assert "Quantum Fixture Note" in titles
    assert all(document.raw_sha256 and document.normalized_text_sha256 for document in documents)


def test_invalid_sidecar_warns_or_errors(tmp_path: Path) -> None:
    source = tmp_path / "bad.md"
    source.write_text("This source is valid text.", encoding="utf-8")
    source.with_suffix(".meta.json").write_text('{"title": "Bad", "extra": true}', encoding="utf-8")

    _documents, warnings, _skipped = read_source_documents(source, _config(source, tmp_path))
    assert warnings[0].code == "invalid_sidecar"

    with pytest.raises(ReaderError, match="invalid sidecar metadata"):
        read_source_documents(source, _config(source, tmp_path, policy="error"))
