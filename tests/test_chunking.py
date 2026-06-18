from __future__ import annotations

from pathlib import Path

from hcaps.ingest.chunking import chunk_document
from hcaps.ingest.readers import read_source_documents
from hcaps.substrate.manifest import SubstrateBuildConfig

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "source_docs"


def test_chunking_is_deterministic_and_preserves_sections(tmp_path: Path) -> None:
    config = SubstrateBuildConfig(
        input_path=FIXTURE_DOCS,
        output_path=tmp_path / "capsules.jsonl",
        manifest_path=tmp_path / "manifest.json",
        max_chunk_chars=240,
    )
    documents, _warnings, _skipped = read_source_documents(FIXTURE_DOCS / "physics_note.md", config)
    first = chunk_document(documents[0], config)
    second = chunk_document(documents[0], config)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].section_path == ["Quantum Note"]
    assert all(chunk.start_char < chunk.end_char for chunk in first)
