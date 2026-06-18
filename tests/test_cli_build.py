from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from hcaps.cli import app

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "source_docs"


def test_cli_build_success_path(tmp_path: Path) -> None:
    output = tmp_path / "capsules.jsonl"
    manifest = tmp_path / "build_manifest.json"
    result = CliRunner().invoke(
        app,
        [
            "build-substrate",
            "--input",
            str(FIXTURE_DOCS),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--cutoff-date",
            "2026-01-01",
            "--max-chunk-chars",
            "280",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert manifest.exists()
    assert "Capsules" in result.output


def test_cli_missing_input_failure(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "build-substrate",
            "--input",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "capsules.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
    )

    assert result.exit_code != 0
    assert "input path does not exist" in result.output
