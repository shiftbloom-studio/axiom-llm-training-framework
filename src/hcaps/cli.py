"""Axiom command-line interface."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated

import orjson
import typer
from rich.console import Console
from rich.table import Table

from hcaps.falsification.audits import audit_input_capsules, findings_to_dicts, summarize_findings
from hcaps.falsification.reports import write_falsification_report
from hcaps.falsification.runner import (
    FalsificationRunConfig,
    load_capsules,
    run_falsification_harness,
)
from hcaps.format.package import create_package_skeleton, inspect_package, validate_package
from hcaps.format.streams import validate_axc_stream
from hcaps.store.jsonl import JsonlCapsuleStore
from hcaps.substrate.builder import build_substrate
from hcaps.substrate.manifest import SubstrateBuildConfig

app = typer.Typer(help="Axiom claim-field substrate tools.")
format_app = typer.Typer(help="AXF/AXC format commands.")
package_app = typer.Typer(help="AXP package commands.")
falsify_app = typer.Typer(help="Generate and audit falsification harness artifacts.")
app.add_typer(format_app, name="format")
app.add_typer(package_app, name="package")
app.add_typer(falsify_app, name="falsify")
console = Console()


@app.command("build-substrate")
def build_substrate_command(
    input_path: Annotated[Path, typer.Option("--input", help="Source file or directory.")],
    output_path: Annotated[Path, typer.Option("--output", help="Output capsules JSONL path.")],
    manifest_path: Annotated[
        Path,
        typer.Option("--manifest", help="Output build manifest JSON path."),
    ],
    cutoff_date: Annotated[
        str | None,
        typer.Option("--cutoff-date", help="Strict temporal cutoff date in YYYY-MM-DD form."),
    ] = None,
    max_chunk_chars: Annotated[int, typer.Option("--max-chunk-chars", min=200)] = 1200,
    family_threshold: Annotated[
        float,
        typer.Option("--family-threshold", min=0.0, max=1.0),
    ] = 0.72,
    axc_output_path: Annotated[
        Path | None,
        typer.Option("--axc-output", help="Canonical AXC capsule stream path."),
    ] = None,
    axp_package_path: Annotated[
        Path | None,
        typer.Option("--package", help="Optional AXP package directory path."),
    ] = None,
) -> None:
    """Build validated claim capsules from local source documents."""

    if not input_path.exists():
        raise typer.BadParameter(f"input path does not exist: {input_path}")

    parsed_cutoff = _parse_cutoff_date(cutoff_date)
    config = SubstrateBuildConfig(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        cutoff_date=parsed_cutoff,
        max_chunk_chars=max_chunk_chars,
        claim_family_similarity_threshold=family_threshold,
        axc_output_path=axc_output_path,
        axp_package_path=axp_package_path,
    )
    try:
        result = build_substrate(config)
    except Exception as exc:
        console.print(f"[red]substrate build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Claim-Field Substrate Build")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Source documents", str(result.manifest.source_document_count))
    table.add_row("Chunks", str(result.manifest.chunk_count))
    table.add_row("Candidate claims", str(result.manifest.candidate_claim_count))
    table.add_row("Claim families", str(result.manifest.claim_family_count))
    table.add_row("Relations", str(result.manifest.relation_candidate_count))
    table.add_row("Capsules", str(result.manifest.emitted_capsule_count))
    table.add_row("Warnings", str(len(result.manifest.warnings)))
    console.print(table)
    console.print(f"Wrote capsules: {output_path}")
    axc_path = axc_output_path or output_path.with_name(f"{output_path.stem}.axc")
    console.print(f"Wrote AXC stream: {axc_path}")
    console.print(f"Wrote manifest: {manifest_path}")
    if axp_package_path is not None:
        console.print(f"Wrote AXP package: {axp_package_path}")


@app.command("inspect-substrate")
def inspect_substrate_command(
    input_path: Annotated[Path, typer.Option("--input", help="Input capsules JSONL path.")],
) -> None:
    """Inspect a built claim-capsule JSONL file."""

    if not input_path.exists():
        raise typer.BadParameter(f"input path does not exist: {input_path}")

    capsules = list(JsonlCapsuleStore().read_capsules(input_path))
    relation_count = sum(len(capsule.relations) for capsule in capsules)
    table = Table(title="Claim-Field Substrate")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Capsules", str(len(capsules)))
    table.add_row("Relations", str(relation_count))
    table.add_row("Provenance records", str(sum(len(capsule.provenance) for capsule in capsules)))
    console.print(table)


def _parse_cutoff_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter("--cutoff-date must use YYYY-MM-DD") from exc


@format_app.command("validate")
def format_validate_command(
    input_path: Annotated[Path, typer.Argument(help="AXC stream path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON report.")] = False,
) -> None:
    """Validate an AXC capsule stream."""

    report = validate_axc_stream(input_path)
    _print_validation_report(report.model_dump(mode="json"), json_output=json_output)
    if not report.ok:
        raise typer.Exit(code=1)


@package_app.command("init")
def package_init_command(
    package_path: Annotated[Path, typer.Argument(help="AXP package directory path.")],
    name: Annotated[str, typer.Option("--name", help="Dataset name.")],
    license_name: Annotated[str, typer.Option("--license", help="Dataset license.")] = "unknown",
) -> None:
    """Create an empty AXP package skeleton."""

    manifest = create_package_skeleton(package_path, dataset_name=name, license=license_name)
    console.print(f"Created AXP package: {package_path}")
    console.print(f"Package ID: {manifest.package_id}")


@package_app.command("validate")
def package_validate_command(
    package_path: Annotated[Path, typer.Argument(help="AXP package directory path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON report.")] = False,
) -> None:
    """Validate an AXP package layout and manifest hashes."""

    report = validate_package(package_path)
    _print_validation_report(report.model_dump(mode="json"), json_output=json_output)
    if not report.ok:
        raise typer.Exit(code=1)


@package_app.command("inspect")
def package_inspect_command(
    package_path: Annotated[Path, typer.Argument(help="AXP package directory path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Inspect an AXP package."""

    summary = inspect_package(package_path)
    if json_output:
        console.print(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode("utf-8"))
        return
    table = Table(title="AXP Package")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)


@falsify_app.command("run")
def falsify_run_command(
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
def falsify_audit_command(
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
        console.print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    console.print(
        "Audit findings: "
        f"{summary.get('error', 0)} errors, "
        f"{summary.get('warning', 0)} warnings, "
        f"{summary.get('info', 0)} info"
    )
    for finding in findings:
        console.print(f"- [{finding.severity}] {finding.audit}: {finding.message}")


@falsify_app.command("report")
def falsify_report_command(
    manifest_path: Annotated[Path, typer.Argument(help="Falsification manifest JSON path.")],
    output: Annotated[Path, typer.Option("--output", help="Markdown report output path.")],
) -> None:
    """Render a Markdown report from a falsification manifest."""

    try:
        write_falsification_report(manifest_path, output)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(output))


def _print_validation_report(payload: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        console.print(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))
        return
    if payload["ok"]:
        console.print("[green]valid[/green]")
        return
    console.print("[red]invalid[/red]")
    issues = payload.get("issues", [])
    if isinstance(issues, list):
        for issue in issues:
            console.print(f"- {issue}")


if __name__ == "__main__":
    app()
