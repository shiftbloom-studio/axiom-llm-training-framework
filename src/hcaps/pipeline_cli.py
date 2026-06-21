"""First-class staged pipeline CLI: harvest -> prepare -> train (arm E).

This module is CLI WIRING ONLY. It reuses the existing pipeline primitives
(build_corpus, compile_axt, AxiomTrainer, the integrated llama-server) without
modifying them, and the training stage always runs arm E with in-model LEARNED
geometry via the verified pioneer config + the trainer guard — the structured-
native HKR hypothesis is never reduced here.

Design notes:
- Integrated llama-server only (no Perplexity).
- Each category asks the must-have settings, then offers an advanced pass; skip
  uses good defaults.
- Prompt LABELS are simplified, self-explaining 1-3 word names; the underlying
  config keys are unchanged.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIONEER_E_CONFIG = PROJECT_ROOT / "configs" / "training" / "pioneer_e_structured_native_geometry.yaml"
HARVEST_ROOT = PROJECT_ROOT / "data" / "harvest"
PREPARED_ROOT = PROJECT_ROOT / "artifacts" / "axt"

console = Console()


# --------------------------------------------------------------------------------------
# Prompt helpers (interactive CLI selections — not LLM prompts)
# --------------------------------------------------------------------------------------
def _heading(text: str) -> None:
    console.print(f"\n[bold cyan]{text}[/bold cyan]")


def _advanced(category: str) -> bool:
    """Return True if the user wants to tune advanced settings for this category."""
    return bool(
        typer.confirm(f"  {category}: advanced setup? (No = sensible defaults)", default=False)
    )


def _ask(label: str, default: Any | None = None, *, hide: bool = False) -> str:
    return str(typer.prompt(f"  {label}", default=default, hide_input=hide)).strip()


def _ask_int(label: str, default: int) -> int:
    while True:
        raw = _ask(label, default)
        try:
            return int(raw)
        except ValueError:
            console.print("  [red]enter a whole number[/red]")


def _ask_float(label: str, default: float) -> float:
    while True:
        raw = _ask(label, default)
        try:
            return float(raw)
        except ValueError:
            console.print("  [red]enter a number[/red]")


def _choice(label: str, options: list[str], default: str) -> str:
    opts = "/".join(options)
    while True:
        raw = _ask(f"{label} ({opts})", default).lower()
        if raw in options:
            return raw
        console.print(f"  [red]pick one of: {opts}[/red]")


# --------------------------------------------------------------------------------------
# Stage 1 — harvest (collect source documents)
# --------------------------------------------------------------------------------------
def _safe_name(n: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in n)[:40]


def run_harvest(*, interactive: bool = True) -> Path:
    """Collect source documents into a folder usable as the prepare input.

    Backends: 'local' (point at an existing folder), 'brave' (web search),
    'fetch' (download given URLs), 'firecrawl' (scrape). Web backends need their
    own API key, prompted here so there is no required pre-setup.
    """
    _heading("Harvest — gather source documents")
    backend = _choice("search backend", ["local", "brave", "fetch", "firecrawl"], "local")

    if backend == "local":
        folder = Path(_ask("source folder", "examples/corpus/ml_software_benchmarks/sources"))
        folder = folder.expanduser().resolve()
        if not folder.exists():
            console.print(f"  [red]folder not found: {folder}[/red]")
            raise typer.Exit(1)
        console.print(f"  [green]using local sources:[/green] {folder}")
        return folder

    out_dir = HARVEST_ROOT / f"harvest_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if backend == "fetch":
        urls = [u.strip() for u in _ask("source URLs (comma-separated)").split(",") if u.strip()]
        _fetch_urls(urls, out_dir)
    elif backend == "brave":
        query = _ask("search query")
        limit = _ask_int("source limit", 8)
        key = os.environ.get("BRAVE_API_KEY") or _ask("Brave API key", "", hide=True)
        if not key:
            console.print("  [red]a Brave API key is required for web search[/red]")
            raise typer.Exit(1)
        _brave_search(query, limit, key, out_dir)
    else:  # firecrawl
        targets = [u.strip() for u in _ask("URLs to scrape (comma-separated)").split(",") if u.strip()]
        key = os.environ.get("FIRECRAWL_API_KEY") or _ask("Firecrawl API key", "", hide=True)
        _firecrawl_scrape(targets, key, out_dir)

    count = len(list(out_dir.glob("*.md")))
    console.print(f"  [green]harvested {count} documents ->[/green] {out_dir}")
    if count == 0:
        console.print("  [yellow]nothing harvested; prepare will have no sources[/yellow]")
    return out_dir


def _write_doc(out_dir: Path, name: str, title: str, body: str, meta: dict[str, Any]) -> None:
    (out_dir / f"{name}.md").write_text(
        f"---\n{json.dumps(meta, indent=2)}\n---\n\n# {title}\n\n{body}\n", encoding="utf-8"
    )


def _brave_search(query: str, limit: int, key: str, out_dir: Path) -> None:
    params = urllib.parse.urlencode({"q": query or " ", "count": min(limit, 20)})
    req = urllib.request.Request(  # noqa: S310
        f"https://api.search.brave.com/res/v1/web/search?{params}",
        headers={"Accept": "application/json", "X-Subscription-Token": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]brave search failed: {exc}[/red]")
        return
    results = (data.get("web", {}) or {}).get("results", [])[:limit]
    for i, w in enumerate(results):
        title = w.get("title", f"result-{i}")
        url = w.get("url", "")
        _write_doc(
            out_dir,
            f"brave_{i:02d}_{_safe_name(title)}",
            title,
            f"{w.get('description', '')}\n\nSource: {url}",
            {"title": title, "source_url": url, "provider": "brave", "query": query},
        )


def _fetch_urls(urls: list[str], out_dir: Path) -> None:
    for i, u in enumerate(urls):
        try:
            with urllib.request.urlopen(u, timeout=15) as r:  # noqa: S310
                body = r.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [yellow]skip {u}: {exc}[/yellow]")
            continue
        _write_doc(
            out_dir, f"fetch_{i:02d}", u, body[:20000], {"source_url": u, "provider": "fetch"}
        )


def _firecrawl_scrape(targets: list[str], key: str, out_dir: Path) -> None:
    if not key:
        console.print("  [red]a Firecrawl API key is required[/red]")
        return
    try:
        from firecrawl import FirecrawlApp  # type: ignore  # noqa: PLC0415
    except ImportError:
        console.print("  [red]firecrawl-py not installed (uv pip install firecrawl-py)[/red]")
        return
    app = FirecrawlApp(api_key=key)
    for i, t in enumerate(targets):
        try:
            res = app.scrape_url(t, params={"formats": ["markdown"]})
            body = (res or {}).get("markdown", "") if isinstance(res, dict) else str(res)
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [yellow]skip {t}: {exc}[/yellow]")
            continue
        _write_doc(out_dir, f"firecrawl_{i:02d}", t, body[:20000], {"source_url": t, "provider": "firecrawl"})


# --------------------------------------------------------------------------------------
# Stage 2 — prepare (corpus build + extraction via integrated llama-server + AXT compile)
# --------------------------------------------------------------------------------------
def _integrated_provider(server: Any) -> Any:
    """Perplexity-free provider config: the local llama-server is the only extractor."""
    from hcaps.providers.config import (  # noqa: PLC0415
        ProviderCacheConfig,
        ProviderCascadeConfig,
        ProviderEndpointConfig,
        ProviderIngressConfig,
    )

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
        escalation=None,
        cascade=ProviderCascadeConfig(enabled=False),
        cache=ProviderCacheConfig(root_path=(PROJECT_ROOT / ".cache/axiom/providers").resolve()),
    )


def run_prepare(*, sources: Path | None = None) -> Path:
    """Build the corpus + compile the AXT bundle from source documents.

    Returns the path to the compiled .axt bundle (the input to training).
    """
    from hcaps.axt.config import AxtCompileConfig  # noqa: PLC0415
    from hcaps.axt.compiler import compile_axt  # noqa: PLC0415
    from hcaps.corpus.builder import build_corpus  # noqa: PLC0415
    from hcaps.corpus.manifest import CorpusBuildConfig  # noqa: PLC0415
    from hcaps.providers.llama_server_runtime import (  # noqa: PLC0415
        LlamaServerConfig,
        start_llama_server,
    )

    _heading("Prepare — extract claims & compile the dataset")
    if sources is None:
        sources = Path(_ask("source folder", "examples/corpus/ml_software_benchmarks/sources"))
    sources = sources.expanduser().resolve()
    if not sources.exists():
        console.print(f"  [red]source folder not found: {sources}[/red]")
        raise typer.Exit(1)

    # must-have: the integrated llama-server extraction model
    model = _ask("extraction model (Hugging Face GGUF repo[:quant] or local .gguf path)")
    if not model:
        console.print("  [red]an extraction model is required[/red]")
        raise typer.Exit(1)
    hf_repo: str | None = None
    local_gguf: str | None = None
    if model.endswith(".gguf") or model.startswith("/") or model.startswith("~"):
        local_gguf = str(Path(model).expanduser())
    else:
        hf_repo = model

    # advanced corpus settings (skip = defaults)
    cutoff_str = "2026-01-01"
    max_chunk_chars = 420
    claim_sim = 0.72
    if _advanced("corpus"):
        cutoff_str = _ask("cutoff date (YYYY-MM-DD)", cutoff_str)
        max_chunk_chars = _ask_int("chunk size (chars)", max_chunk_chars)
        claim_sim = _ask_float("claim similarity", claim_sim)
    try:
        cutoff: date | None = date.fromisoformat(cutoff_str) if cutoff_str else None
    except ValueError:
        console.print("  [red]cutoff date must be YYYY-MM-DD[/red]")
        raise typer.Exit(1) from None

    run_name = f"prepared_{int(time.time())}"
    run_dir = PROJECT_ROOT / "runs" / run_name
    corpus_out = run_dir / "corpus"
    corpus_out.mkdir(parents=True, exist_ok=True)

    console.print(f"  [bold]starting integrated llama-server[/bold] ({model})")
    log_path = run_dir / "llama_server" / "llama-server.log"
    with start_llama_server(
        LlamaServerConfig(
            hf_repo=hf_repo,
            local_gguf_path=local_gguf,
            log_path=log_path,
            status_callback=lambda m: console.print(f"  [dim]{m}[/dim]"),
        )
    ) as server:
        providers = _integrated_provider(server)
        corpus_cfg = CorpusBuildConfig(
            corpus_id=f"staged.{run_name}.v0",
            dataset_name=run_name,
            input_path=sources,
            output_dir=corpus_out,
            cutoff_date=cutoff,
            max_chunk_chars=max_chunk_chars,
            claim_family_similarity_threshold=claim_sim,
            providers=providers,
        )
        console.print("  [bold]building corpus (extracting claims)…[/bold]")
        corpus_result = build_corpus(corpus_cfg)
    console.print(
        f"  sources={corpus_result.manifest.source_count} "
        f"capsules={corpus_result.manifest.capsule_count}"
    )

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
    console.print("  [bold]compiling AXT tensor bundle…[/bold]")
    axt_res = compile_axt(axt_cfg, force=True)
    console.print(f"  [green]dataset ready:[/green] {axt_path} ({axt_res.manifest.record_count} records)")
    return axt_path


# --------------------------------------------------------------------------------------
# Stage 3 — train (arm E: structured-native + in-model LEARNED geometry)
# --------------------------------------------------------------------------------------
def run_train(*, dataset: Path | None = None) -> Path:
    """Train arm E on a compiled .axt bundle, with learned in-model geometry.

    Uses the verified pioneer config (geometry_provider_injected + the trainer
    guard), so the HKR structured-native hypothesis is never silently reduced.
    """
    from hcaps.training.config import TrainingConfig  # noqa: PLC0415
    from hcaps.training.trainer import AxiomTrainer  # noqa: PLC0415

    _heading("Train — arm E (structured-native + learned geometry)")
    if not PIONEER_E_CONFIG.exists():
        console.print(f"  [red]pioneer E config missing: {PIONEER_E_CONFIG}[/red]")
        raise typer.Exit(1)

    if dataset is None:
        dataset = Path(_ask("dataset (compiled .axt bundle)", "artifacts/axt/pioneer.axt"))
    dataset = dataset.expanduser()

    steps = _ask_int("training steps", 7500)

    seed, precision, output_dir, lr, batch = 13, "fp32", "runs", 0.0003, 8
    if _advanced("training"):
        seed = _ask_int("random seed", seed)
        precision = _choice("precision", ["fp32", "bf16"], precision)
        output_dir = _ask("output folder", output_dir)
        lr = _ask_float("learning rate", lr)
        batch = _ask_int("batch size", batch)

    cfg = yaml.safe_load(PIONEER_E_CONFIG.read_text(encoding="utf-8")) or {}
    run_name = f"e_run_{int(time.time())}"
    cfg.update(
        {
            "run_name": run_name,
            "input_axt_path": str(dataset),
            "max_steps": int(steps),
            "seed": int(seed),
            "precision": precision,
            "output_dir": output_dir,
            "learning_rate": float(lr),
            "micro_batch_size": int(batch),
            "global_batch_size": int(batch),
            "gradient_accumulation_steps": 1,
        }
    )
    budget = dict(cfg.get("compute_budget_config") or {})
    budget["max_train_steps"] = int(steps)
    cfg["compute_budget_config"] = budget

    derived = PROJECT_ROOT / "runs" / run_name / "e_run.yaml"
    derived.parent.mkdir(parents=True, exist_ok=True)
    derived.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    console.print(f"  [bold]training arm E[/bold] on {dataset} for {steps} steps ({precision})")
    # TrainingConfig validation + the trainer guard enforce geometry_provider_injected.
    result = AxiomTrainer(TrainingConfig.model_validate(cfg)).fit()
    console.print(f"  [green]done — checkpoints:[/green] {result.arm_dir}/checkpoints/")
    return Path(result.run_dir)


# --------------------------------------------------------------------------------------
# Full — chain the three stages
# --------------------------------------------------------------------------------------
def run_full() -> None:
    _heading("Axiom — full pipeline (harvest → prepare → train arm E)")
    console.print(
        "Integrated llama-server only. Each stage asks the essentials; advanced is optional."
    )
    if typer.confirm("\nCollect source documents now (harvest)?", default=False):
        sources = run_harvest()
    else:
        sources = Path(_ask("source folder", "examples/corpus/ml_software_benchmarks/sources"))
    dataset = run_prepare(sources=sources)
    run_train(dataset=dataset)
    console.print("\n[bold green]Full pipeline complete.[/bold green]")
