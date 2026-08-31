"""Axiom command-line interface."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import orjson
import typer
import yaml
from rich.console import Console
from rich.table import Table

from hcaps import pipeline_cli
from hcaps.axt import (
    AxtBatchCollator,
    AxtCompileConfig,
    AxtDataset,
    compile_axt,
    validate_axt_bundle,
)
from hcaps.axt.inspect import compare_axt_bundles, inspect_axt_bundle, inspect_tensor_group
from hcaps.corpus.builder import build_corpus
from hcaps.corpus.gold import export_gold_candidates
from hcaps.corpus.manifest import CorpusBuildConfig, CorpusBuildResult, load_corpus_build_config
from hcaps.experiments.arms import default_arm_catalog
from hcaps.experiments.comparison import control_effects, pairwise_metric_deltas
from hcaps.experiments.configs import ExperimentSuiteConfig
from hcaps.experiments.orchestrator import ExperimentOrchestrator
from hcaps.falsification.audits import audit_input_capsules, findings_to_dicts, summarize_findings
from hcaps.falsification.reports import write_falsification_report
from hcaps.falsification.runner import (
    FalsificationRunConfig,
    load_capsules,
    run_falsification_harness,
)
from hcaps.format.package import create_package_skeleton, inspect_package, validate_package
from hcaps.format.streams import validate_axc_stream
from hcaps.geometry import GeometryConfig
from hcaps.geometry.cli import (
    compute_reference_geometry,
    inspect_axt_geometry,
    sample_axt_loops,
    smoke_geometry,
)
from hcaps.model import AxiomModelConfig, AxiomStructuredModel
from hcaps.model.parameter_count import parameter_count
from hcaps.operator import export_run, inspect_run, list_runs
from hcaps.providers.cache import ProviderCache
from hcaps.providers.config import (
    ProviderCacheConfig,
    ProviderCascadeConfig,
    ProviderEndpointConfig,
    ProviderGateConfig,
    ProviderIngressConfig,
    load_provider_ingress_config,
)
from hcaps.providers.llama_server_runtime import (
    LlamaServerConfig,
    LlamaServerHandle,
    start_llama_server,
)
from hcaps.scoring import score_run
from hcaps.store.jsonl import JsonlCapsuleStore
from hcaps.substrate.builder import build_substrate
from hcaps.substrate.manifest import SubstrateBuildConfig
from hcaps.training import AxiomTrainer, TrainingConfig
from hcaps.training.checkpointing import load_training_checkpoint
from hcaps.training.config import ComputeBudgetConfig, CurriculumConfig, LossWeights
from hcaps.training.logging import write_json
from hcaps.utils.time import utc_now
from hcaps.verdict import VerdictThresholds, generate_verdict, write_verdict_report
from hcaps.verdict.report import inspect_verdict

app = typer.Typer(help="Axiom claim-field substrate tools.")
format_app = typer.Typer(help="AXF/AXC format commands.")
package_app = typer.Typer(help="AXP package commands.")
falsify_app = typer.Typer(help="Generate and audit falsification harness artifacts.")
corpus_app = typer.Typer(help="Provider-aware corpus ingress commands.")
corpus_providers_app = typer.Typer(help="Provider configuration checks.")
corpus_cache_app = typer.Typer(help="Provider cache inspection.")
corpus_gold_app = typer.Typer(help="Human-review candidate exports.")
axt_app = typer.Typer(help="AXT tensor bundle compiler and runtime interface commands.")
model_app = typer.Typer(help="Structured-native model stack developer commands.")
geometry_app = typer.Typer(help="Learned geometry developer commands.")
train_app = typer.Typer(help="P6 training runtime commands.")
experiment_app = typer.Typer(help="P6 experiment suite commands.")
score_app = typer.Typer(help="P6 scoring commands.")
verdict_app = typer.Typer(help="P6 verdict commands.")
run_app = typer.Typer(help="P6 operator run inspection/export commands.")
app.add_typer(format_app, name="format")
app.add_typer(package_app, name="package")
app.add_typer(falsify_app, name="falsify")
app.add_typer(corpus_app, name="corpus")
app.add_typer(axt_app, name="axt")
app.add_typer(model_app, name="model")
app.add_typer(geometry_app, name="geometry")
app.add_typer(train_app, name="train")
app.add_typer(experiment_app, name="experiment")
app.add_typer(score_app, name="score")
app.add_typer(verdict_app, name="verdict")
app.add_typer(run_app, name="run")
corpus_app.add_typer(corpus_providers_app, name="providers")
corpus_app.add_typer(corpus_cache_app, name="cache")
corpus_app.add_typer(corpus_gold_app, name="gold")
console = Console()

TRAINING_SIZE_PROFILES: dict[str, dict[str, int | float | None]] = {
    "smoke": {
        "max_steps": 2,
        "max_records_seen": 24,
        "max_wall_clock_seconds": 300.0,
    },
    "pilot": {
        "max_steps": 20,
        "max_records_seen": 256,
        "max_wall_clock_seconds": 1800.0,
    },
    "standard": {
        "max_steps": 100,
        "max_records_seen": 2048,
        "max_wall_clock_seconds": 7200.0,
    },
}
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
PERPLEXITY_API_KEY_ENV = "PERPLEXITY_API_KEY"
DEFAULT_PERPLEXITY_MODEL = "sonar-pro"
FULL_PIPELINE_HIGH_IMPACT_CLAIM_TYPES = [
    "causal_claim",
    "measurement_claim",
    "method_claim",
    "scientific_claim",
]


@app.command("full")
def full_command() -> None:
    """Run the complete staged pipeline: harvest -> prepare -> train arm E."""

    pipeline_cli.run_full()


@app.command("harvest")
def harvest_command() -> None:
    """Collect source documents (local folder, Brave search, fetch URLs, or Firecrawl)."""

    pipeline_cli.run_harvest()


@app.command("prepare")
def prepare_command() -> None:
    """Extract claims with the integrated llama-server and compile the AXT dataset."""

    pipeline_cli.run_prepare()


def _run_full_pipeline_wizard() -> None:
    console.print("[bold cyan]Axiom Full Pipeline[/bold cyan]")
    console.print(
        "Runs provider-backed data extraction, AXT compilation, the complete P6 arm catalog,\n"
        "scoring, verdict generation, and reproducibility artifacts."
    )
    sources_path = _prompt_source_directory()
    cutoff = _parse_cutoff_date(
        typer.prompt("Temporal cutoff date (YYYY-MM-DD)", default="2026-01-01")
    )
    training_profile = _prompt_training_size()
    run_name = _default_full_run_name(training_profile)
    trainer_template = _training_template_for_profile(training_profile)
    extraction_mode = _prompt_extraction_mode()

    try:
        if extraction_mode == "integrated":
            final_dir = _run_full_pipeline_with_integrated_llama(
                run_name=run_name,
                sources_path=sources_path,
                cutoff=cutoff,
                trainer_template=trainer_template,
            )
        else:
            providers = _prompt_remote_openai_provider_config()
            final_dir = _execute_full_pipeline(
                run_name=run_name,
                sources_dir=sources_path,
                cutoff=cutoff,
                providers=providers,
                trainer_template=trainer_template,
            )
    except Exception as exc:
        console.print(f"[red]Full pipeline failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_pipeline_success(final_dir, run_name)


def _prompt_source_directory() -> Path:
    default_sources = Path("examples/corpus/ml_software_benchmarks/sources")
    sources_str = typer.prompt(
        "Source documents directory",
        default=str(default_sources),
    )
    sources_path = Path(sources_str).expanduser().resolve()
    if not sources_path.exists():
        console.print(f"[red]Source path does not exist: {sources_path}[/red]")
        raise typer.Exit(1)
    return sources_path


def _prompt_training_size() -> str:
    raw = str(
        typer.prompt(
            "Training size (smoke, pilot, standard)",
            default="smoke",
        )
    ).strip()
    aliases: dict[str, str] = {
        "s": "smoke",
        "p": "pilot",
        "std": "standard",
        "full": "standard",
    }
    profile = aliases.get(raw.casefold(), raw.casefold())
    if profile not in TRAINING_SIZE_PROFILES:
        console.print("[red]Training size must be smoke, pilot, or standard.[/red]")
        raise typer.Exit(1)
    return profile


def _prompt_extraction_mode() -> str:
    raw = str(
        typer.prompt(
            "Data extraction provider (integrated or remote)",
            default="integrated",
        )
    ).strip()
    mode = raw.casefold()
    if mode not in {"integrated", "remote"}:
        console.print("[red]Extraction provider must be integrated or remote.[/red]")
        raise typer.Exit(1)
    return mode


def _default_full_run_name(training_profile: str) -> str:
    stamp = utc_now().strftime("%Y%m%d_%H%M%S")
    return f"full_{training_profile}_{stamp}"


def _training_template_for_profile(profile: str) -> TrainingConfig:
    settings = TRAINING_SIZE_PROFILES[profile]
    base = _default_smoke_trainer_template()
    compute_budget = base.compute_budget_config.model_copy(
        update={
            "max_train_steps": int(settings["max_steps"] or 1),
            "max_records_seen": settings["max_records_seen"],
            "max_wall_clock_seconds": settings["max_wall_clock_seconds"],
        }
    )
    payload = base.model_dump(mode="python")
    payload.update(
        {
            "max_steps": int(settings["max_steps"] or 1),
            "compute_budget_config": compute_budget,
        }
    )
    return TrainingConfig.model_validate(payload)


def _run_full_pipeline_with_integrated_llama(
    *,
    run_name: str,
    sources_path: Path,
    cutoff: date | None,
    trainer_template: TrainingConfig,
) -> Path:
    hf_repo = typer.prompt(
        "Hugging Face GGUF model for data extraction (repo[:quant])",
    ).strip()
    if not hf_repo:
        console.print("[red]A Hugging Face GGUF model is required for integrated extraction.[/red]")
        raise typer.Exit(1)
    perplexity = _prompt_perplexity_endpoint_config(
        provider_id="perplexity_sonar_escalation",
        prompt_label="Perplexity escalation model",
    )
    log_path = Path("runs") / run_name / "llama_server" / "llama-server.log"
    run_dir, wizard_dir = _prepare_full_pipeline_run(run_name, Path("runs"))
    console.print("[bold]Starting integrated llama-server for corpus extraction[/bold]")
    with start_llama_server(
        LlamaServerConfig(
            hf_repo=hf_repo,
            log_path=log_path,
            status_callback=lambda message: console.print(f"[dim]{message}[/dim]"),
        )
    ) as server:
        providers = _integrated_llama_provider_config(server, escalation=perplexity)
        corpus_result = _build_full_pipeline_corpus(
            run_name=run_name,
            sources_dir=sources_path,
            cutoff=cutoff,
            providers=providers,
            run_dir=run_dir,
            wizard_dir=wizard_dir,
        )
    console.print("[bold]Integrated llama-server stopped after corpus extraction[/bold]")
    return _execute_full_pipeline_after_corpus(
        run_name=run_name,
        corpus_result=corpus_result,
        trainer_template=trainer_template,
        output_root=Path("runs"),
        run_dir=run_dir,
        wizard_dir=wizard_dir,
    )


def _integrated_llama_provider_config(
    server: LlamaServerHandle,
    *,
    escalation: ProviderEndpointConfig,
) -> ProviderIngressConfig:
    env_name = "AXIOM_LLAMA_SERVER_API_KEY"
    os.environ[env_name] = server.api_key
    primary = ProviderEndpointConfig(
        provider_id="integrated_llama_server",
        type="openai_compatible",
        provider_family="llama.cpp",
        provider_mode="local",
        model=server.model_name,
        base_url=server.base_url,
        api_key_env=env_name,
        timeout_seconds=server.config.request_timeout_seconds,
        max_retries=1,
        dry_run=False,
    )
    return ProviderIngressConfig(
        primary=primary,
        escalation=escalation,
        cascade=_full_pipeline_cascade_config(),
        cache=ProviderCacheConfig(
            root_path=Path(".cache/axiom/providers").resolve(),
            mode="live",
        ),
    )


def _prompt_remote_openai_provider_config() -> ProviderIngressConfig:
    primary = _prompt_perplexity_endpoint_config(
        provider_id="perplexity_sonar_primary",
        prompt_label="Perplexity extraction model",
    )
    return ProviderIngressConfig(
        primary=primary,
        escalation=None,
        cascade=ProviderCascadeConfig(enabled=False),
        cache=ProviderCacheConfig(
            root_path=Path(".cache/axiom/providers").resolve(),
            mode="live",
        ),
    )


def _prompt_perplexity_endpoint_config(
    *,
    provider_id: str,
    prompt_label: str,
) -> ProviderEndpointConfig:
    model = str(
        typer.prompt(
            prompt_label,
            default=DEFAULT_PERPLEXITY_MODEL,
        )
    ).strip()
    if not model:
        console.print("[red]Perplexity model is required.[/red]")
        raise typer.Exit(1)
    api_key_env = _ensure_perplexity_api_key()
    return _perplexity_endpoint_config(
        provider_id=provider_id,
        model=model,
        api_key_env=api_key_env,
    )


def _ensure_perplexity_api_key() -> str:
    if os.environ.get(PERPLEXITY_API_KEY_ENV):
        console.print(f"[green]Using ${PERPLEXITY_API_KEY_ENV} for Perplexity API calls.[/green]")
        return PERPLEXITY_API_KEY_ENV
    api_key = typer.prompt(
        "Perplexity API key (hidden; stored only for this process)",
        hide_input=True,
    ).strip()
    if not api_key:
        console.print("[red]Perplexity API key is required for cascade escalation.[/red]")
        raise typer.Exit(1)
    os.environ[PERPLEXITY_API_KEY_ENV] = api_key
    return PERPLEXITY_API_KEY_ENV


def _perplexity_endpoint_config(
    *,
    provider_id: str,
    model: str,
    api_key_env: str,
) -> ProviderEndpointConfig:
    return ProviderEndpointConfig(
        provider_id=provider_id,
        type="openai_compatible",
        provider_family="perplexity_sonar",
        provider_mode="remote",
        model=model,
        base_url=PERPLEXITY_BASE_URL,
        api_key_env=api_key_env,
        timeout_seconds=180.0,
        max_retries=2,
        dry_run=False,
    )


def _full_pipeline_cascade_config() -> ProviderCascadeConfig:
    return ProviderCascadeConfig(
        enabled=True,
        gate=ProviderGateConfig(
            min_confidence=0.72,
            high_impact_claim_types=FULL_PIPELINE_HIGH_IMPACT_CLAIM_TYPES,
            max_escalation_fraction=1.0,
            escalate_on_warnings=True,
        ),
        merge_policy="schema_grounded_confidence_priority_v0.1",
    )


def _print_pipeline_success(final_dir: Path, run_name: str) -> None:
    console.print("\n[bold green]Pipeline complete — trained model checkpoints ready.[/bold green]")
    console.print(f"  Run dir: {final_dir}")
    console.print(f"  Model checkpoints: {final_dir}/arms/*/checkpoints/latest.pt")
    console.print(f"  Verdict report: {final_dir}/verdict/report.md")
    console.print(
        f"  Repro export: axiom run export {final_dir} --output artifacts/{run_name}-repro.json"
    )


@app.command("build-substrate")
def build_substrate_command(  # noqa: PLR0917
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
def axt_compile_command(  # noqa: PLR0917
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


@model_app.command("config-summary")
def model_config_summary_command(
    config_path: Annotated[Path, typer.Argument(help="P4 model YAML config path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Inspect a structured-native model config without constructing training state."""

    try:
        config = AxiomModelConfig.from_yaml(config_path)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = {
        "model_dim": config.model_dim,
        "slot_dim": config.slot_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "parameter_budget_hint": config.parameter_budget_hint,
        "enabled_modules": config.enabled_module_flags(),
        "active_ablation_modes": config.ablations.active,
        "text_projection": config.effective_text_projection(),
        "geometry_mode": config.effective_geometry_mode(),
        "router_mode": config.effective_router_mode(),
    }
    _print_payload(payload, json_output=json_output, title="Axiom Model Config")


@model_app.command("count-params")
def model_count_params_command(
    config_path: Annotated[Path, typer.Argument(help="P4 model YAML config path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Construct the P4 model stack and count parameters."""

    try:
        config = AxiomModelConfig.from_yaml(config_path)
        model = AxiomStructuredModel(config)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = {
        **parameter_count(model),
        "parameter_budget_hint": config.parameter_budget_hint,
        "p4_model_schema_version": config.p4_model_schema_version,
    }
    _print_payload(payload, json_output=json_output, title="Axiom Model Parameters")


@model_app.command("smoke-forward")
def model_smoke_forward_command(
    bundle_path: Annotated[Path, typer.Argument(help="AXT bundle directory.")],
    config_path: Annotated[
        Path,
        typer.Option("--config", help="P4 model YAML config path."),
    ],
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 2,
    split_name: Annotated[str | None, typer.Option("--split-name")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Run one no-training forward pass over a P3 AXT bundle."""

    try:
        dataset = AxtDataset(bundle_path, split_name=split_name)
        if len(dataset) == 0:
            raise ValueError("AXT dataset is empty")
        records = [dataset[index] for index in range(min(batch_size, len(dataset)))]
        batch = AxtBatchCollator()(records)
        config = AxiomModelConfig.from_yaml(config_path)
        model = AxiomStructuredModel(config)
        output = model(batch)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc

    payload = {
        "batch_size": output.latent_state.shape[0],
        "latent_shape": tuple(output.latent_state.shape),
        "slot_shape": tuple(output.slot_states.shape),
        "text_projection_shape": tuple(output.text_projection_logits.shape)
        if output.text_projection_logits is not None
        else None,
        "axc_out_head_shapes": output.raw_axc_out.head_shapes(),
        "router": output.router_diagnostics,
        "geometry": output.geometry_diagnostics,
    }
    _print_payload(payload, json_output=json_output, title="Axiom Model Smoke Forward")


@geometry_app.command("inspect-axt")
def geometry_inspect_axt_command(
    bundle_path: Annotated[Path, typer.Argument(help="AXT bundle directory.")],
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 4,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Inspect geometry graph structure derived from an AXT bundle."""

    try:
        payload = inspect_axt_geometry(bundle_path, batch_size=batch_size)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(payload, json_output=json_output, title="Axiom Geometry AXT Inspect")


@geometry_app.command("sample-loops")
def geometry_sample_loops_command(
    bundle_path: Annotated[Path, typer.Argument(help="AXT bundle directory.")],
    output: Annotated[Path, typer.Option("--output", help="Loop sample JSON output path.")],
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Geometry YAML config path."),
    ] = Path("configs/geometry/geometry_learned_smoke.yaml"),
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 4,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Sample bounded geometry loops from AXT-derived context transitions."""

    try:
        config = GeometryConfig.from_yaml(config_path)
        payload = sample_axt_loops(bundle_path, output=output, config=config, batch_size=batch_size)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(payload, json_output=json_output, title="Axiom Geometry Loops")


@geometry_app.command("compute-reference")
def geometry_compute_reference_command(
    bundle_path: Annotated[Path, typer.Argument(help="AXT bundle directory.")],
    output: Annotated[Path, typer.Option("--output", help="Reference JSON output path.")],
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Geometry YAML config path."),
    ] = Path("configs/geometry/geometry_reference_smoke.yaml"),
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 4,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Compute off-loop reference geometry diagnostics for a tiny AXT batch."""

    try:
        config = GeometryConfig.from_yaml(config_path)
        payload = compute_reference_geometry(
            bundle_path,
            output=output,
            config=config,
            batch_size=batch_size,
        )
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(payload, json_output=json_output, title="Axiom Geometry Reference")


@geometry_app.command("smoke")
def geometry_smoke_command(
    bundle_path: Annotated[Path, typer.Argument(help="AXT bundle directory.")],
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Geometry YAML config path."),
    ] = Path("configs/geometry/geometry_learned_smoke.yaml"),
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 4,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Run a no-training learned-geometry smoke forward over an AXT bundle."""

    try:
        config = GeometryConfig.from_yaml(config_path)
        payload = smoke_geometry(bundle_path, config=config, batch_size=batch_size)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(payload, json_output=json_output, title="Axiom Geometry Smoke")


@train_app.callback(invoke_without_command=True)
def train_default(ctx: typer.Context) -> None:
    """Interactive arm-E training (structured-native + learned geometry).

    `axiom train` with no subcommand runs the guided arm-E run; `axiom train run
    <config>` still runs a config directly.
    """
    if ctx.invoked_subcommand is None:
        pipeline_cli.run_train()


@train_app.command("run")
def train_run_command(
    config_path: Annotated[Path, typer.Argument(help="P6 training YAML config path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Run one P6 training arm from an AXT bundle."""

    try:
        result = AxiomTrainer(TrainingConfig.from_yaml(config_path)).fit()
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = {
        "run_id": result.run_id,
        "arm_id": result.arm_id,
        "run_dir": str(result.run_dir),
        "checkpoint": str(result.checkpoint_path),
        "metrics": str(result.metrics_path),
        "manifest": str(result.manifest_path),
        "final_step": result.final_step,
    }
    _print_payload(payload, json_output=json_output, title="Axiom P6 Training")


@train_app.command("resume")
def train_resume_command(
    checkpoint_path: Annotated[Path, typer.Argument(help="P6 training checkpoint path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Resume a P6 training arm from a checkpoint that carries its config."""

    try:
        payload = load_training_checkpoint(checkpoint_path)
        config_payload = payload.get("training_config")
        if not isinstance(config_payload, dict):
            raise ValueError("checkpoint does not contain a replayable training_config")
        config = TrainingConfig.model_validate(config_payload).model_copy(
            update={"resume_from": checkpoint_path, "overwrite": True}
        )
        result = AxiomTrainer(config).fit()
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(
        {
            "run_id": result.run_id,
            "arm_id": result.arm_id,
            "resumed_from": str(checkpoint_path),
            "final_step": result.final_step,
        },
        json_output=json_output,
        title="Axiom P6 Resume",
    )


@train_app.command("inspect")
def train_inspect_command(
    run_dir: Annotated[Path, typer.Argument(help="P6 run directory.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Inspect a P6 training run directory."""

    try:
        payload = inspect_run(run_dir)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(payload, json_output=json_output, title="Axiom P6 Run")


@experiment_app.command("plan")
def experiment_plan_command(
    config_path: Annotated[Path, typer.Argument(help="P6 experiment suite YAML config path.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional copy of the expanded plan JSON."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Expand a P6 experiment suite without training arms."""

    try:
        plan = ExperimentOrchestrator(ExperimentSuiteConfig.from_yaml(config_path)).prepare()
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(plan.plan_path.read_bytes())
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(
        {
            "run_id": plan.run_id,
            "run_dir": str(plan.run_dir),
            "arms": [arm.arm_id for arm in plan.arms],
            "plan_path": str(plan.plan_path),
        },
        json_output=json_output,
        title="Axiom P6 Experiment Plan",
    )


@experiment_app.command("run")
def experiment_run_command(
    config_path: Annotated[Path, typer.Argument(help="P6 experiment suite YAML config path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Run a P6 experiment suite sequentially without external services."""

    try:
        result = ExperimentOrchestrator(ExperimentSuiteConfig.from_yaml(config_path)).run()
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(
        {
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "arms": list(result.arm_results),
            "scores": str(result.scoring_result.scores_path),
            "verdict": str(result.verdict_path),
            "fairness_report": str(result.fairness_report_path),
        },
        json_output=json_output,
        title="Axiom P6 Experiment Run",
    )


@experiment_app.command("score")
def experiment_score_command(
    run_dir: Annotated[Path, typer.Argument(help="P6 run directory.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Score a completed P6 run."""

    try:
        result = score_run(run_dir)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(
        {"run_dir": str(result.run_dir), "scores_path": str(result.scores_path)},
        json_output=json_output,
        title="Axiom P6 Score",
    )


@experiment_app.command("compare")
def experiment_compare_command(
    run_dir: Annotated[Path, typer.Argument(help="P6 run directory.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON comparison.")] = False,
) -> None:
    """Compute pairwise metrics and control effects for a scored P6 run."""

    try:
        result = score_run(run_dir)
        payload: dict[str, object] = {
            "pairwise_metrics": pairwise_metric_deltas(result.scores),
            "control_effects": control_effects(result.scores),
        }
        comparisons_dir = run_dir / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        pairwise = payload["pairwise_metrics"]
        controls = payload["control_effects"]
        write_json(
            comparisons_dir / "pairwise_metrics.json",
            pairwise if isinstance(pairwise, dict) else {},
        )
        write_json(
            comparisons_dir / "control_effects.json",
            controls if isinstance(controls, dict) else {},
        )
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(payload, json_output=json_output, title="Axiom P6 Compare")


@score_app.command("run")
def score_run_command(
    run_dir: Annotated[Path, typer.Argument(help="P6 run directory.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Score all P6 arms in a run directory."""

    try:
        result = score_run(run_dir)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(result.scores, json_output=json_output, title="Axiom P6 Scores")


@verdict_app.command("report")
def verdict_report_command(
    run_dir: Annotated[Path, typer.Argument(help="P6 run directory.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Markdown verdict report output path."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional verdict thresholds YAML."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Generate a P6 verdict report from scored artifacts."""

    try:
        scores = score_run(run_dir).scores
        fairness_path = run_dir / "comparisons" / "fairness_report.json"
        fairness = json.loads(fairness_path.read_text("utf-8")) if fairness_path.exists() else {}
        thresholds = VerdictThresholds.from_yaml(config_path) if config_path is not None else None
        verdict = generate_verdict(scores, fairness_report=fairness, thresholds=thresholds)
        report_path = write_verdict_report(run_dir, verdict=verdict, output=output)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(
        {
            "run_dir": str(run_dir),
            "report": str(report_path),
            "overall": verdict["overall_verdict"],
        },
        json_output=json_output,
        title="Axiom P6 Verdict",
    )


@verdict_app.command("inspect")
def verdict_inspect_command(
    verdict_path: Annotated[Path, typer.Argument(help="P6 verdict JSON path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON verdict.")] = False,
) -> None:
    """Inspect a P6 verdict JSON file."""

    try:
        payload = inspect_verdict(verdict_path)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(payload, json_output=json_output, title="Axiom P6 Verdict Inspect")


@run_app.command("inspect")
def run_inspect_command(
    run_dir: Annotated[Path, typer.Argument(help="P6 run directory.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Inspect a P6 run directory."""

    try:
        payload = inspect_run(run_dir)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(payload, json_output=json_output, title="Axiom P6 Run")


@run_app.command("export")
def run_export_command(
    run_dir: Annotated[Path, typer.Argument(help="P6 run directory.")],
    output: Annotated[Path, typer.Option("--output", help="Reproducibility bundle path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summary.")] = False,
) -> None:
    """Export a P6 reproducibility bundle."""

    try:
        exported = export_run(run_dir, output=output)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(
        {"run_dir": str(run_dir), "export": str(exported)},
        json_output=json_output,
        title="Axiom P6 Export",
    )


@run_app.command("list")
def run_list_command(
    runs_dir: Annotated[Path, typer.Argument(help="Directory containing P6 runs.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON summaries.")] = False,
) -> None:
    """List P6 run directories."""

    try:
        payload: dict[str, object] = {"runs": list_runs(runs_dir)}
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_payload(payload, json_output=json_output, title="Axiom P6 Runs")


@run_app.callback(invoke_without_command=True)
def run_root(ctx: typer.Context) -> None:
    """P6 operator run inspection/export commands.

    Bare 'axiom run' (no subcommand) launches the interactive guided full-pipeline trainer.
    It starts at corpus ingress (prompts only for required values: sources dir, optional
    provider base URL + API key when chosen, run name) and drives the complete chain
    (no reductions): provider-aware corpus -> full AXT (neighborhoods, negatives, geometry
    slots, provider context, masks, all targets) -> AXP/AXT -> full experiment suite using
    the complete P6 arm catalog (text baselines + structured-native no-geo + geo
    + ablation/control arms) + scoring + verdict + reproducibility artifacts.

    All other configuration uses complete smoke-scale defaults from the P6 handoff. The
    resulting runs/<name>/ contains trained checkpoints (latest.pt per arm) and is fully
    compatible with 'axiom run inspect' etc.
    """
    if ctx.invoked_subcommand is None:
        _run_interactive_training_pipeline()


def _run_interactive_training_pipeline() -> None:
    """Collect minimal required prompts then drive the full unreduced pipeline."""
    console.print("[bold cyan]Axiom Interactive Full Pipeline[/bold cyan]")
    console.print(
        "Guides from corpus ingress (OpenAI-compat IP+key optional) to finished\n"
        "trained checkpoints. Enter accepts defaults. Only prompted for values\n"
        "that must be set (no safe default)."
    )

    default_sources = Path("examples/corpus/ml_software_benchmarks/sources")
    sources_str = typer.prompt(
        "Source documents directory for corpus ingress",
        default=str(default_sources),
    )
    sources_path = Path(sources_str).resolve()
    if not sources_path.exists():
        console.print(f"[red]Source path does not exist: {sources_path}[/red]")
        raise typer.Exit(1)

    cutoff_str = typer.prompt("Temporal cutoff date (YYYY-MM-DD)", default="2026-01-01")
    cutoff = _parse_cutoff_date(cutoff_str)

    providers = _prompt_provider_ingress_config()

    run_name = typer.prompt("Training run name (required; output under runs/<name>)").strip()
    if not run_name:
        console.print("[red]Run name is required and cannot be empty.[/red]")
        raise typer.Exit(1)
    if any(ch in run_name for ch in '\\/:*?"<>|'):
        console.print("[yellow]Warning: run name contains filesystem-unsafe characters.[/yellow]")

    ready = typer.confirm(
        f"Run FULL unreduced pipeline for '{run_name}' "
        f"({'provider' if providers else 'deterministic'})?\n"
        "  - corpus build (full claim-field, traces, negatives, gold hooks)\n"
        "  - AXT compile (all tensor groups, masks, neighborhoods, geometry slots,\n"
        "    provider ctx)\n"
        "  - experiment (complete arm set: text baselines + structured + learned\n"
        "    geometry + controls)\n"
        "  - scoring + verdict + manifests + repro bundle\n"
        "All first-class must-haves and ablations preserved. Smoke-scale for speed.",
        default=True,
    )
    if not ready:
        console.print("Aborted by user.")
        raise typer.Exit(0)

    try:
        final_dir = _execute_full_pipeline(
            run_name=run_name,
            sources_dir=sources_path,
            cutoff=cutoff,
            providers=providers,
        )
        console.print(
            "\n[bold green]Pipeline complete — trained model checkpoints ready.[/bold green]"
        )
        console.print(f"  Run dir: {final_dir}")
        console.print(
            f"  Model checkpoints: {final_dir}/arms/*/checkpoints/latest.pt (and step_*.pt)"
        )
        console.print(f"  Verdict report: {final_dir}/verdict/report.md")
        console.print("\nInspect / export with the operator surface:")
        console.print(f"  ./bin/axiom run inspect {final_dir}")
        console.print(
            f"  ./bin/axiom run export {final_dir} --output artifacts/{run_name}-repro.json"
        )
    except Exception as exc:
        console.print(f"[red]Pipeline failed:[/red] {exc}")
        raise typer.Exit(1) from exc


def _prompt_provider_ingress_config() -> ProviderIngressConfig | None:
    use_provider = typer.confirm(
        "Use OpenAI-compatible provider for extraction (supply base URL + API key)?\n"
        "  No = deterministic bootstrap (no key, fully offline, cached-safe).",
        default=False,
    )
    if not use_provider:
        return None

    base_url = typer.prompt(
        "Provider base URL (e.g. http://localhost:8000/v1 or remote OpenAI-compat)",
        default="http://localhost:8000/v1",
    )
    provider_mode = typer.prompt(
        "Provider mode (local or remote)",
        default="local",
    ).strip()
    if provider_mode not in {"local", "remote"}:
        console.print("[red]Provider mode must be 'local' or 'remote'.[/red]")
        raise typer.Exit(1)
    model = typer.prompt("Extraction model name", default="local-extractor-model")
    api_key = typer.prompt("Provider API Key (hidden)", hide_input=True).strip()
    if not api_key:
        console.print("[red]API key required for provider mode.[/red]")
        raise typer.Exit(1)
    env_name = typer.prompt(
        "Env var name to hold the key (set in-process only, never written to files)",
        default="AXIOM_PROVIDER_API_KEY",
    )
    os.environ[env_name] = api_key
    primary = ProviderEndpointConfig(
        provider_id="interactive_openai_compatible",
        type="openai_compatible",
        provider_family="openai_compatible",
        provider_mode=provider_mode,
        model=model,
        base_url=base_url,
        api_key_env=env_name,
        dry_run=False,
    )
    console.print(f"[green]Provider active (key held in ${env_name} for this run only).[/green]")
    return ProviderIngressConfig(
        primary=primary,
        escalation=None,
        cascade=ProviderCascadeConfig(enabled=False),
        cache=ProviderCacheConfig(
            root_path=Path(".cache/axiom/providers").resolve(),
            mode="live",
        ),
    )


def _execute_full_pipeline(
    run_name: str,
    sources_dir: Path,
    cutoff: date | None,
    providers: ProviderIngressConfig | None,
    *,
    arms: list[str] | None = None,
    trainer_template: TrainingConfig | None = None,
    output_root: Path = Path("runs"),
) -> Path:
    """Execute the corpus->AXT->full experiment pipeline. Snapshots all generated configs.

    Test callers may pass arms= reduced list and a tiny trainer_template to keep pytest fast.
    The interactive path always uses the complete P6 arm catalog + smoke hyperparams.
    """
    run_dir, wizard_dir = _prepare_full_pipeline_run(run_name, output_root)
    corpus_result = _build_full_pipeline_corpus(
        run_name=run_name,
        sources_dir=sources_dir,
        cutoff=cutoff,
        providers=providers,
        run_dir=run_dir,
        wizard_dir=wizard_dir,
    )
    return _execute_full_pipeline_after_corpus(
        run_name=run_name,
        corpus_result=corpus_result,
        trainer_template=trainer_template,
        output_root=output_root,
        run_dir=run_dir,
        wizard_dir=wizard_dir,
        arms=arms,
    )


def _prepare_full_pipeline_run(
    run_name: str,
    output_root: Path,
) -> tuple[Path, Path]:
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    wizard_dir = run_dir / "wizard_configs"
    wizard_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, wizard_dir


def _build_full_pipeline_corpus(
    *,
    run_name: str,
    sources_dir: Path,
    cutoff: date | None,
    providers: ProviderIngressConfig | None,
    run_dir: Path,
    wizard_dir: Path,
) -> CorpusBuildResult:
    # 1. Corpus
    corpus_out = run_dir / "corpus"
    corpus_out.mkdir(parents=True, exist_ok=True)
    corpus_cfg = CorpusBuildConfig(
        corpus_id=f"interactive.{run_name}.v0",
        dataset_name=run_name,
        input_path=sources_dir,
        output_dir=corpus_out,
        cutoff_date=cutoff,
        providers=providers,
        # defaults for chunking, threshold, strict temporal, no pdf
    )
    corpus_cfg.to_yaml(wizard_dir / "corpus.yaml")
    if providers is not None:
        _write_yaml(wizard_dir / "providers.yaml", {"providers": providers.model_dump(mode="json")})

    console.print("[bold]Phase 1/3: Corpus ingress & build[/bold]")
    corpus_result = build_corpus(corpus_cfg)
    console.print(
        f"  Sources: {corpus_result.manifest.source_count}  "
        f"Capsules: {corpus_result.manifest.capsule_count}  "
        f"AXP: {corpus_result.package_path}"
    )
    return corpus_result


def _execute_full_pipeline_after_corpus(
    *,
    run_name: str,
    corpus_result: CorpusBuildResult,
    trainer_template: TrainingConfig | None,
    output_root: Path,
    run_dir: Path,
    wizard_dir: Path,
    arms: list[str] | None = None,
) -> Path:
    # 2. AXT (explicit for control; full include flags)
    axt_path = run_dir / "axt" / f"{run_name}.axt"
    axt_cfg = AxtCompileConfig(
        input_path=corpus_result.package_path,
        output_path=axt_path,
        allow_all_without_split=True,
        max_text_length=128,
        include_provider_context=True,
        include_relation_neighborhoods=True,
        include_negative_samples=True,
        include_geometry_slots=True,
        include_evaluation_references=False,
        strict_temporal_masks=True,
        strict_schema_validation=True,
        hash_artifacts=True,
        seed=13,
    )
    axt_cfg.write_json(wizard_dir / "axt_compile.json")

    console.print("[bold]Phase 2/3: AXT compile (full tensor interface)[/bold]")
    axt_res = compile_axt(axt_cfg, force=True)
    console.print(f"  Records: {axt_res.manifest.record_count}  Bundle: {axt_path}")

    # 3. Experiment (unreduced arm catalog + smoke hyperparams)
    console.print("[bold]Phase 3/3: Experiment orchestration + training + scoring + verdict[/bold]")
    tmpl = trainer_template if trainer_template is not None else _default_smoke_trainer_template()
    use_arms = list(arms) if arms is not None else _full_p6_arm_ids()

    suite = ExperimentSuiteConfig(
        suite_name=run_name,
        input_axp=None,
        input_axt=axt_path,
        output_dir=output_root,
        arms=use_arms,
        seeds=[13],
        budget_profile="smoke",
        trainer_config_template=tmpl,
        model_config_template=Path("configs/model/structured_native_smoke.yaml"),
        geometry_config_template=Path("configs/geometry/geometry_learned_smoke.yaml"),
        scoring_config=Path("configs/scoring/default_structured_scoring.yaml"),
        verdict_config=Path("configs/verdict/conservative_thresholds.yaml"),
        source_content_id=run_name,
        split_id="all_without_split",
    )
    suite.to_yaml(wizard_dir / "suite.yaml")

    orchestrator = ExperimentOrchestrator(suite)
    exp_res = orchestrator.run()
    return exp_res.run_dir


def _write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump"):
        payload: dict[str, Any] = data.model_dump(mode="json")
    else:
        payload = data
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _full_p6_arm_ids() -> list[str]:
    """Return the complete default P6 experiment arm set."""
    return list(
        default_arm_catalog(
            input_axt_path="",
            model_config="",
            source_content_id="interactive",
            seed=13,
        ).keys()
    )


def _default_smoke_trainer_template() -> TrainingConfig:
    """Full smoke-scale TrainingConfig (hyperparams, weights, curriculum, budget).

    No reductions from the P6 smoke defaults.
    """
    loss_w = LossWeights(
        structured_axc_out=1.0,
        relation_prediction=1.0,
        provenance_recovery=1.0,
        epistemic_proxy=1.0,
        stability_temporal=1.0,
        uncertainty_calibration=0.5,
        geometry_observables=0.2,
        geometry_regularization=0.01,
        text_projection=0.2,
    )
    curric = CurriculumConfig(
        schedule_name="none",
        phase_boundaries=[0, 1, 1, 1, 1, 2],
        text_projection_weight_schedule=[1.0],
        structured_loss_weight_schedule=[1.0],
        relation_neighborhood_depth_schedule=[1],
        side_channel_dropout_schedule=[0.0],
        geometry_activation_schedule=[1.0],
        context_dropout_schedule=[0.0],
        negative_sample_hardness_schedule=[0.0],
    )
    budget = ComputeBudgetConfig(
        max_train_steps=2,
        max_records_seen=24,
        max_wall_clock_seconds=300.0,
        parameter_match_tolerance=0.25,
        compute_match_tolerance=0.25,
        estimate_flops=True,
    )
    return TrainingConfig(
        run_name="placeholder",
        seed=13,
        input_axt_path=Path("placeholder.axt"),
        output_dir=Path("runs"),
        model_config_path=Path("configs/model/structured_native_smoke.yaml"),
        arm_name="placeholder",
        max_steps=2,
        max_epochs=1,
        global_batch_size=2,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=0.001,
        weight_decay=0.0,
        optimizer="adamw",
        scheduler="none",
        warmup_steps=0,
        cooldown_steps=0,
        clip_grad_norm=1.0,
        precision="fp32",
        checkpoint_interval=1,
        eval_interval=1,
        log_interval=1,
        save_optimizer_state=True,
        resume_from=None,
        text_projection_loss_weight=0.2,
        geometry_loss_weight=0.2,
        structured_loss_weights=loss_w,
        curriculum_config=curric,
        compute_budget_config=budget,
        geometry_config_path=Path("configs/geometry/geometry_learned_smoke.yaml"),
        geometry_provider_kind="learned",
        control_transform=None,
        text_mode=None,
        overwrite=True,
    )


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
def falsify_run_command(  # noqa: PLR0917
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
