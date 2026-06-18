from __future__ import annotations

from pathlib import Path

from hcaps.extraction.claims import extract_claims
from hcaps.extraction.relations import generate_relation_candidates
from hcaps.ingest.chunking import chunk_documents
from hcaps.ingest.readers import read_source_documents
from hcaps.schema.capsule import RelationType
from hcaps.substrate.builder import build_claim_families
from hcaps.substrate.manifest import SubstrateBuildConfig

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "source_docs"


def test_relation_candidate_generation(tmp_path: Path) -> None:
    config = SubstrateBuildConfig(
        input_path=FIXTURE_DOCS,
        output_path=tmp_path / "capsules.jsonl",
        manifest_path=tmp_path / "manifest.json",
        claim_family_similarity_threshold=0.9,
    )
    documents, _warnings, _skipped = read_source_documents(FIXTURE_DOCS / "physics_note.md", config)
    claims = extract_claims(chunk_documents(documents, config), documents, config)
    families = build_claim_families(claims, documents, config)

    relations = generate_relation_candidates(families, claims)

    assert relations
    assert {relation.relation_type for relation in relations} & {
        RelationType.SUPPORTS,
        RelationType.CONTRADICTS,
        RelationType.RELATED,
    }
    assert all(relation.relation_id.startswith("rel_") for relation in relations)
