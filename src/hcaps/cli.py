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

from hcaps.axt import AxtCompileConfig, compile_axt, validate_axt_bundle
from hcaps.axt.inspect import compare_axt_bundles, inspect_axt_bundle, inspect_tensor_group
from hcaps.corpus.builder import build_corpus
from hcaps.corpus.gold import export_gold_candidates
from hcaps.corpus.manifest import load_corpus_build_config
from hcaps.falsification.audits import audit_input_capsules, findings_to_dicts, summarize_findings
from hcaps.falsification.reports import write_falsification_report
from hcaps.falsification.runner import (
    FalsificationRunConfig,
    load_capsules,
    run_falsification_harness,
)
from hcaps.format.package import create_package_skeleton, inspect_package, validate_package
from hcaps.format.streams import validate_axc_stream
from hcaps.providers.cache import ProviderCache
from hcaps.providers.config import load_provider_ingress_config
from hcaps.store.jsonl import JsonlCapsuleStore
from hcaps.substrate.builder import build_substrate
from hcaps.substrate.manifest import SubstrateBuildConfig

app = typer.Typer(help="Axiom claim-field substrate tools.")
format_app = typer.Typer(help="AXF/AXC format commands.")
package_app = typer.Typer(help="AXP package commands.")
falsify_app = typer.Typer(help="Generate and audit falsification harness artifacts.")
corpus_app = typer.Typer(help="Provider-aware corpus ingress commands.")
corpus_providers_app = typer.Typer(help="Provider configuration checks.")
corpus_cache_app = typer.Typer(help="Provider cache inspection.")
corpus_gold_app = typer.Typer(help="Human-review candidate exports.")
axt_app = typer.Typer(help="AXT tensor bundle compiler and runtime interface commands.")
app.add_typer(format_app, name="format")
app.add_typer(package_app, name="package")
app.add_typer(falsify_app, name="falsify")
app.add_typer(corpus_app, name="corpus")
app.add_typer(axt_app, name="axt")
corpus_app.add_typer(corpus_providers_app, name="providers")
corpus_app.add_typer(corpus_cache_app, name="cache")
corpus_app.add_typer(corpus_gold_app, name="gold")
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


@corpus_app.command("build")
def corpus_build_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Corpus YAML config path."),
    ],
) -> None:
    """Build a provider-aware Axiom corpus artifact set."""

    try:
        config = load_corpus_build_config(config_path)
        result = build_corpus(config)
    except Exception as exc:
        console.print(f"[red]corpus build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Axiom Corpus Build")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Sources", str(result.manifest.source_count))
    table.add_row("Claim families", str(result.manifest.claim_family_count))
    table.add_row("Capsules", str(result.manifest.capsule_count))
    table.add_row("Relations", str(result.manifest.relation_candidate_count))
    table.add_row("Warnings", str(len(result.manifest.warnings)))
    console.print(table)
    console.print(f"Wrote corpus manifest: {result.artifact_paths['corpus_manifest']}")
    console.print(f"Wrote AXP package: {result.package_path}")


@corpus_app.command("inspect")
def corpus_inspect_command(
    input_path: Annotated[Path, typer.Argument(help="Corpus manifest or AXP package path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Inspect a corpus manifest or generated AXP package."""

    if not input_path.exists():
        raise typer.BadParameter(f"input path does not exist: {input_path}")
    if input_path.is_file():
        payload = orjson.loads(input_path.read_bytes())
    else:
        payload = inspect_package(input_path)
    if json_output:
        console.print(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))
        return
    table = Table(title="Axiom Corpus")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(str(key), str(value))
    console.print(table)


@corpus_providers_app.command("check")
def corpus_providers_check_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Provider YAML config path."),
    ],
) -> None:
    """Validate provider ingress configuration without making network calls."""

    try:
        config = load_provider_ingress_config(config_path)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="Provider Ingress")
    table.add_column("Role")
    table.add_column("Provider ID")
    table.add_column("Type")
    table.add_column("Mode")
    table.add_column("Model")
    table.add_row(
        "primary",
        config.primary.provider_id,
        config.primary.type,
        config.primary.provider_mode,
        config.primary.model,
    )
    if config.escalation is not None:
        table.add_row(
            "escalation",
            config.escalation.provider_id,
            config.escalation.type,
            config.escalation.provider_mode,
            config.escalation.model,
        )
    console.print(table)
    console.print(f"Cascade enabled: {config.cascade.enabled}")
    console.print(f"Cache root: {config.cache.root_path}")


@corpus_cache_app.command("inspect")
def corpus_cache_inspect_command(
    cache_path: Annotated[Path, typer.Argument(help="Provider cache root path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Inspect provider cache records without reading secrets."""

    manifest = ProviderCache(cache_path).manifest()
    payload = manifest.model_dump(mode="json")
    if json_output:
        console.print(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))
        return
    table = Table(title="Provider Cache")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Root", payload["root_path"])
    table.add_row("Records", str(payload["record_count"]))
    table.add_row("Provider fingerprints", ", ".join(payload["provider_fingerprints"]))
    console.print(table)


@corpus_gold_app.command("export")
def corpus_gold_export_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", help="AXP package or AXC stream input path."),
    ],
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Gold-reference candidate JSONL output path."),
    ],
) -> None:
    """Export human-review candidate records."""

    try:
        exported = export_gold_candidates(input_path, output_path)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Wrote gold-reference candidates: {exported}")


def _parse_cutoff_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter("--cutoff-date must use YYYY-MM-DD") from exc


@axt_app.command("compile")
def axt_compile_command(
    input_path: Annotated[Path, typer.Option("--input", help="AXC stream or AXP package.")],
    output_path: Annotated[Path, typer.Option("--output", help="Output .axt bundle directory.")],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional AXT YAML/JSON compile config."),
    ] = None,
    split_name: Annotated[str | None, typer.Option("--split-name")] = None,
    allow_all_without_split: Annotated[
        bool,
        typer.Option("--allow-all-without-split/--require-split"),
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing bundle.")] = False,
) -> None:
    """Compile AXC/AXP into an AXT tensor bundle."""

    try:
        config = (
            AxtCompileConfig.from_file(config_path)
            if config_path is not None
            else AxtCompileConfig(input_path=input_path, output_path=output_path)
        )
        config = config.model_copy(
            update={
                "input_path": input_path,
                "output_path": output_path,
                "split_name": split_name if split_name is not None else config.split_name,
                "allow_all_without_split": allow_all_without_split
                or config.allow_all_without_split,
            }
        )
        result = compile_axt(config, force=force)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title="AXT Compile")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Output", str(result.output_path))
    table.add_row("Records", str(result.manifest.record_count))
    table.add_row("Source format", result.manifest.source_format)
    table.add_row("Tensor groups", str(len(result.manifest.tensor_groups)))
    table.add_row("Warnings", str(len(result.manifest.warnings)))
    console.print(table)


@axt_app.command("inspect")
def axt_inspect_command(
    bundle_path: Annotated[Path, typer.Argument(help="AXT bundle directory.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Inspect an AXT bundle manifest."""

    try:
        payload = inspect_axt_bundle(bundle_path)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(payload, json_output=json_output, title="AXT Bundle")


@axt_app.command("validate")
def axt_validate_command(
    bundle_path: Annotated[Path, typer.Argument(help="AXT bundle directory.")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Print JSON validation report.")
    ] = False,
) -> None:
    """Validate an AXT bundle manifest, hashes, and required tensor groups."""

    report = validate_axt_bundle(bundle_path)
    _print_validation_report(report, json_output=json_output)
    if not report["ok"]:
        raise typer.Exit(code=1)


@axt_app.command("tensor")
def axt_tensor_command(
    bundle_path: Annotated[Path, typer.Argument(help="AXT bundle directory.")],
    group: Annotated[str, typer.Option("--group", help="Tensor group name.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON tensor summary.")] = False,
) -> None:
    """Inspect one AXT tensor group."""

    try:
        payload = inspect_tensor_group(bundle_path, group)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(payload, json_output=json_output, title=f"AXT Tensor Group: {group}")


@axt_app.command("compare")
def axt_compare_command(
    left_bundle: Annotated[Path, typer.Argument(help="Left AXT bundle directory.")],
    right_bundle: Annotated[Path, typer.Argument(help="Right AXT bundle directory.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON comparison.")] = False,
) -> None:
    """Compare two AXT bundles by manifest artifact hashes."""

    try:
        payload = compare_axt_bundles(left_bundle, right_bundle)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(payload, json_output=json_output, title="AXT Compare")


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


def _print_payload(payload: dict[str, object], *, json_output: bool, title: str) -> None:
    if json_output:
        console.print(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))
        return
    table = Table(title=title)
    table.add_column("Field")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(str(key), str(value))
    console.print(table)


if __name__ == "__main__":
    app()
