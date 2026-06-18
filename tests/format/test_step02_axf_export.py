from __future__ import annotations

from pathlib import Path

from hcaps.format.package import validate_package
from hcaps.format.streams import read_axc_stream, validate_axc_stream
from hcaps.substrate.builder import ClaimFieldSubstrateBuilder
from hcaps.substrate.manifest import SubstrateBuildConfig

FIXTURE_DOCS = Path(__file__).parents[1] / "fixtures" / "source_docs"


def test_step02_builder_exports_valid_axc_and_axp(tmp_path: Path) -> None:
    output = tmp_path / "capsules.jsonl"
    axc_output = tmp_path / "capsules.axc"
    package_path = tmp_path / "fixture.axp"
    config = SubstrateBuildConfig(
        input_path=FIXTURE_DOCS,
        output_path=output,
        axc_output_path=axc_output,
        axp_package_path=package_path,
        manifest_path=tmp_path / "build_manifest.json",
        cutoff_date="2026-01-01",
        max_chunk_chars=280,
    )

    result = ClaimFieldSubstrateBuilder(config).build()
    axc_report = validate_axc_stream(axc_output)
    package_report = validate_package(package_path)
    axc_capsules = list(read_axc_stream(axc_output))

    assert result.axc_capsules
    assert axc_report.ok
    assert package_report.ok
    assert len(axc_capsules) == result.manifest.emitted_capsule_count
    assert str(axc_output) in result.manifest.output_file_hashes
