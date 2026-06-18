"""Command-line interface for Axiom."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from hcaps.falsification.audits import audit_input_capsules, findings_to_dicts, summarize_findings
from hcaps.falsification.reports import write_falsification_report
from hcaps.falsification.runner import (
    FalsificationRunConfig,
    load_capsules,
    run_falsification_harness,
)

app = typer.Typer(help="Axiom claim-centric pretraining framework.")
falsify_app = typer.Typer(help="Generate and audit falsification harness artifacts.")
app.add_typer(falsify_app, name="falsify")


@falsify_app.command("run")
def falsify_run(
    input_path: Annotated[Path, typer.Argument(help="AXC file or AXP package path.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", help="Output directory root.")],
    arms: Annotated[
        str,
        typer.Option("--arms", help="Comma-separated arm names or 'all'."),
    ] = "all",
    seed: Annotated[int, typer.Option("--seed", help="Deterministic seed.")] = 13,
    tokenizer_path: Annotated[
        Path | None,
        typer.Option("--tokenizer-path", help="Optional WhitespaceTokenizer JSON path."),
    ] = None,
    max_length: Annotated[int | None, typer.Option("--max-length")] = None,
    include_side_channels: Annotated[
        bool,
        typer.Option("--include-side-channels/--no-side-channels"),
    ] = True,
    strict_audits: Annotated[bool, typer.Option("--strict-audits/--no-strict-audits")] = False,
    split_name: Annotated[str | None, typer.Option("--split-name")] = None,
    allow_all_without_split: Annotated[
        bool,
        typer.Option(
            "--allow-all-without-split",
            help="Allow an AXP package to run without a named split.",
        ),
    ] = False,
) -> None:
    """Create falsification artifacts and a run manifest."""

    try:
        result = run_falsification_harness(
            FalsificationRunConfig(
                input_path=input_path,
                output_dir=output_dir,
                arms=arms,
                seed=seed,
                tokenizer_path=tokenizer_path,
                max_length=max_length,
                include_side_channels=include_side_channels,
                strict_audits=strict_audits,
                split_name=split_name,
                allow_all_without_split=allow_all_without_split,
            )
        )
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(result.manifest_path))


@falsify_app.command("audit")
def falsify_audit(
    input_path: Annotated[Path, typer.Argument(help="AXC file or AXP package path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON audit output.")] = False,
) -> None:
    """Run input audits without generating arms."""

    try:
        capsules, source_format, _ = load_capsules(input_path)
        findings = audit_input_capsules(capsules)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = summarize_findings(findings)
    payload = {
        "input_path": str(input_path),
        "source_format": source_format,
        "summary": summary,
        "findings": findings_to_dicts(findings),
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    typer.echo(
        "Audit findings: "
        f"{summary.get('error', 0)} errors, "
        f"{summary.get('warning', 0)} warnings, "
        f"{summary.get('info', 0)} info"
    )
    for finding in findings:
        typer.echo(f"- [{finding.severity}] {finding.audit}: {finding.message}")


@falsify_app.command("report")
def falsify_report(
    manifest_path: Annotated[Path, typer.Argument(help="Falsification manifest JSON path.")],
    output: Annotated[Path, typer.Option("--output", help="Markdown report output path.")],
) -> None:
    """Render a Markdown report from a falsification manifest."""

    try:
        write_falsification_report(manifest_path, output)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(output))


if __name__ == "__main__":
    app()
