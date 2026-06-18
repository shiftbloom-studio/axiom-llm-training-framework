from __future__ import annotations

from pathlib import Path

from hcaps.falsification.reports import write_falsification_report
from hcaps.falsification.runner import FalsificationRunConfig, run_falsification_harness

EXAMPLES = Path("examples/axf/v0_1")


def test_report_generator_writes_markdown_without_result_claims(tmp_path: Path) -> None:
    result = run_falsification_harness(
        FalsificationRunConfig(
            input_path=EXAMPLES / "minimal_dataset.axp",
            output_dir=tmp_path,
            arms=["flat_text", "structured_text", "capsule_text"],
            allow_all_without_split=True,
        )
    )
    report_path = tmp_path / "report.md"

    write_falsification_report(result.manifest, report_path)

    report = report_path.read_text(encoding="utf-8")
    assert "# Axiom Falsification Harness Report" in report
    assert "benchmark" not in report.casefold()
    assert "performance" not in report.casefold()
