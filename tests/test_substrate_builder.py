from __future__ import annotations

from pathlib import Path

from hcaps.store.jsonl import JsonlCapsuleStore
from hcaps.substrate.builder import ClaimFieldSubstrateBuilder, read_build_manifest
from hcaps.substrate.manifest import SubstrateBuildConfig
from hcaps.utils.hashing import file_sha256

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "source_docs"


def test_substrate_builder_emits_valid_capsules_manifest_and_hashes(tmp_path: Path) -> None:
    output = tmp_path / "claim-field" / "capsules.jsonl"
    manifest_path = tmp_path / "claim-field" / "build_manifest.json"
    config = SubstrateBuildConfig(
        input_path=FIXTURE_DOCS,
        output_path=output,
        manifest_path=manifest_path,
        cutoff_date="2026-01-01",
        max_chunk_chars=280,
    )

    result = ClaimFieldSubstrateBuilder(config).build()
    capsules = list(JsonlCapsuleStore().read_capsules(output))
    manifest = read_build_manifest(manifest_path)
    axc_output = output.with_suffix(".axc")

    assert capsules
    assert axc_output.exists()
    assert len(capsules) == result.manifest.emitted_capsule_count
    assert manifest.output_file_hashes[str(output)] == file_sha256(output)
    assert manifest.output_file_hashes[str(axc_output)] == file_sha256(axc_output)
    assert manifest.source_document_count == 3
    assert manifest.skipped_file_count == 1
    assert any(warning.code == "post_cutoff_source_excluded" for warning in manifest.warnings)
    assert all(
        record.source_title != "Future Fixture Note"
        for capsule in capsules
        for record in capsule.provenance
    )


def test_substrate_builder_jsonl_roundtrip_preserves_capsule_ids(tmp_path: Path) -> None:
    output = tmp_path / "capsules.jsonl"
    config = SubstrateBuildConfig(
        input_path=FIXTURE_DOCS / "physics_note.md",
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
    )

    result = ClaimFieldSubstrateBuilder(config).build()
    roundtripped = list(JsonlCapsuleStore().read_capsules(output))

    assert [capsule.capsule_id for capsule in roundtripped] == [
        capsule.capsule_id for capsule in result.capsules
    ]
