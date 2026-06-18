from __future__ import annotations

from pathlib import Path

from hcaps.format.package import validate_package
from hcaps.format.streams import read_axc_stream, validate_axc_stream
from hcaps.format.validation import forbidden_truth_field_issues
from hcaps.substrate.builder import ClaimFieldSubstrateBuilder
from hcaps.substrate.manifest import SubstrateBuildConfig

FIXTURE_DOCS = Path(__file__).parents[1] / "fixtures" / "source_docs"


def test_builder_export_is_axf_conformant_and_cutoff_safe(tmp_path: Path) -> None:
    axc_output = tmp_path / "capsules.axc"
    package_path = tmp_path / "builder.axp"
    config = SubstrateBuildConfig(
        input_path=FIXTURE_DOCS,
        output_path=tmp_path / "legacy_capsules.jsonl",
        axc_output_path=axc_output,
        axp_package_path=package_path,
        manifest_path=tmp_path / "build_manifest.json",
        cutoff_date="2026-01-01",
        max_chunk_chars=280,
    )

    result = ClaimFieldSubstrateBuilder(config).build()
    capsules = list(read_axc_stream(axc_output))

    assert result.axc_capsules
    assert validate_axc_stream(axc_output).ok
    assert validate_package(package_path).ok
    assert len(capsules) == result.manifest.emitted_capsule_count
    for capsule in capsules:
        record = capsule.model_dump(mode="json")
        assert not forbidden_truth_field_issues(record)
        assert capsule.provenance.sources
        for source in capsule.provenance.sources:
            if source.source_date is not None and not source.target_only:
                assert source.source_date <= capsule.temporal.valid_as_of
