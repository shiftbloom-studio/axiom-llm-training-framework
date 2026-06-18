from __future__ import annotations

from pathlib import Path

from hcaps.extraction.claims import DeterministicClaimExtractor
from hcaps.ingest.chunking import chunk_document
from hcaps.ingest.readers import read_source_documents
from hcaps.schema.capsule import ClaimType
from hcaps.substrate.manifest import SubstrateBuildConfig

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "source_docs"


def test_deterministic_claim_extraction(tmp_path: Path) -> None:
    config = SubstrateBuildConfig(
        input_path=FIXTURE_DOCS,
        output_path=tmp_path / "capsules.jsonl",
        manifest_path=tmp_path / "manifest.json",
    )
    documents, _warnings, _skipped = read_source_documents(
        FIXTURE_DOCS / "medicine_note.txt", config
    )
    chunks = chunk_document(documents[0], config)
    extractor = DeterministicClaimExtractor()

    claims = [claim for chunk in chunks for claim in extractor.extract(chunk, documents[0], config)]
    again = [claim for chunk in chunks for claim in extractor.extract(chunk, documents[0], config)]

    assert [claim.claim_id for claim in claims] == [claim.claim_id for claim in again]
    assert any(claim.claim_type == ClaimType.CAUSAL_CLAIM for claim in claims)
    assert all(0.0 <= claim.confidence <= 1.0 for claim in claims)
