from __future__ import annotations

from pathlib import Path

from hcaps.extraction.claims import extract_claims
from hcaps.ingest.chunking import chunk_documents
from hcaps.ingest.readers import read_source_documents
from hcaps.substrate.builder import build_claim_families
from hcaps.substrate.manifest import SubstrateBuildConfig

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "source_docs"


def test_claim_family_grouping_is_deterministic(tmp_path: Path) -> None:
    config = SubstrateBuildConfig(
        input_path=FIXTURE_DOCS,
        output_path=tmp_path / "capsules.jsonl",
        manifest_path=tmp_path / "manifest.json",
        claim_family_similarity_threshold=0.45,
    )
    documents, _warnings, _skipped = read_source_documents(
        FIXTURE_DOCS / "medicine_note.txt", config
    )
    claims = extract_claims(chunk_documents(documents, config), documents, config)

    first = build_claim_families(claims, documents, config)
    second = build_claim_families(claims, documents, config)

    assert [family.family_id for family in first] == [family.family_id for family in second]
    assert any(family.evidence_count >= 2 for family in first)
    assert all(family.schema_claim_id.startswith("claim_") for family in first)
