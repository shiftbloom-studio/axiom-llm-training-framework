from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from hcaps.cli import app


def test_cli_axt_compile_inspect_validate_tensor_and_compare(tmp_path: Path) -> None:
    runner = CliRunner()
    left = tmp_path / "left.axt"
    right = tmp_path / "right.axt"
    minimal_axp = (
        Path(__file__).resolve().parents[2] / "examples" / "axf" / "v0_1" / "minimal_dataset.axp"
    )

    compile_result = runner.invoke(
        app,
        [
            "axt",
            "compile",
            "--input",
            str(minimal_axp),
            "--output",
            str(left),
            "--allow-all-without-split",
        ],
    )
    assert compile_result.exit_code == 0, compile_result.output

    inspect_result = runner.invoke(app, ["axt", "inspect", str(left), "--json"])
    assert inspect_result.exit_code == 0
    assert '"format": "AXT"' in inspect_result.output

    validate_result = runner.invoke(app, ["axt", "validate", str(left)])
    assert validate_result.exit_code == 0

    tensor_result = runner.invoke(app, ["axt", "tensor", str(left), "--group", "claim", "--json"])
    assert tensor_result.exit_code == 0
    assert "claim_type_idx" in tensor_result.output

    runner.invoke(
        app,
        [
            "axt",
            "compile",
            "--input",
            str(minimal_axp),
            "--output",
            str(right),
            "--allow-all-without-split",
        ],
    )
    compare_result = runner.invoke(app, ["axt", "compare", str(left), str(right), "--json"])
    assert compare_result.exit_code == 0
    assert '"left_record_count": 2' in compare_result.output
