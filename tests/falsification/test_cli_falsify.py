from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from hcaps.cli import app

EXAMPLES = Path("examples/axf/v0_1")


def test_cli_audit_works_on_minimal_axc() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["falsify", "audit", str(EXAMPLES / "minimal_capsules.axc"), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["error"] == 0


def test_cli_run_works_on_minimal_axp(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "falsify",
            "run",
            str(EXAMPLES / "minimal_dataset.axp"),
            "--output-dir",
            str(tmp_path),
            "--arms",
            "flat_text,structured_text,capsule_text",
            "--seed",
            "13",
            "--allow-all-without-split",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest_path = Path(result.output.strip())
    assert manifest_path.exists()
