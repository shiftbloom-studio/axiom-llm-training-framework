#!/usr/bin/env python3
"""Axiom optional fine-art abstract web UI server.

This is a completely separate, optional control application. It does not edit,
monkey-patch, or depend on private internals of the core training framework
except through the same public + documented surfaces that the CLI already uses
(CorpusBuildConfig, ProviderIngressConfig, build_corpus, compile_axt,
ExperimentOrchestrator, start_llama_server, etc).

Launch (recommended):
    ./bin/axiom-ui

Or directly:
    uv --python 3.14 --with fastapi --with 'uvicorn[standard]' \
       --with jinja2 --with python-multipart --with 'watchfiles>=0.20' \
       python -m ui.server

The UI owns:
- form-driven generation of corpus + provider + suite configs
- explicit data-volume levers (source sampling, escalation fraction, chunking)
- launching (remote cascade + integrated local P1 w/o remote LLM + auto GGUF) while streaming
- web harvest (Firecrawl/Brave/fetch) producing provenance-rich sources for P1
- live artistic visualization + metric dashboards driven by the same
  artifacts (manifests, jsonl logs, traces) the native pipeline writes
- zero mutation of core source, specs, or default configs
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import suppress
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None  # type: ignore[assignment]

import yaml
from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Project root + path discipline (mirrors bin/axiom, never mutates core)
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC_PATH = PROJECT_ROOT / "src"
RUNS_ROOT = PROJECT_ROOT / "runs"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
UI_DATA_ROOT = HERE / ".ui-data"
UI_DATA_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

UI_STATE_FILE = HERE / ".ui-current-job.json"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# --------------------------------------------------------------------------------------
# Optional deep integration (launch uses real surfaces when available)
# --------------------------------------------------------------------------------------
HAS_CORE: bool = False
build_corpus = None  # type: ignore
compile_axt = None  # type: ignore
ExperimentOrchestrator = None  # type: ignore
ExperimentSuiteConfig = None  # type: ignore
AxtCompileConfig = None  # type: ignore
CorpusBuildConfig = None  # type: ignore
ProviderIngressConfig = None  # type: ignore
ProviderEndpointConfig = None  # type: ignore
ProviderCascadeConfig = None  # type: ignore
ProviderGateConfig = None  # type: ignore
ProviderCacheConfig = None  # type: ignore
LlamaServerConfig = None  # type: ignore
start_llama_server = None  # type: ignore
TrainingConfig = None  # type: ignore
LossWeights = None  # type: ignore
CurriculumConfig = None  # type: ignore
ComputeBudgetConfig = None  # type: ignore
default_arm_catalog = None  # type: ignore

try:
    from hcaps.axt import compile_axt as _compile_axt  # type: ignore
    from hcaps.axt.config import AxtCompileConfig as _AxtCompileConfig  # type: ignore
    from hcaps.corpus.builder import build_corpus as _build_corpus  # type: ignore
    from hcaps.corpus.manifest import CorpusBuildConfig as _CorpusBuildConfig  # type: ignore
    from hcaps.experiments.arms import default_arm_catalog as _default_arm_catalog  # type: ignore
    from hcaps.experiments.configs import (
        ExperimentSuiteConfig as _ExperimentSuiteConfig,  # type: ignore
    )
    from hcaps.experiments.orchestrator import (
        ExperimentOrchestrator as _ExperimentOrchestrator,  # type: ignore
    )
    from hcaps.providers.config import (  # type: ignore
        ProviderCacheConfig as _ProviderCacheConfig,
    )
    from hcaps.providers.config import (
        ProviderCascadeConfig as _ProviderCascadeConfig,
    )
    from hcaps.providers.config import (
        ProviderEndpointConfig as _ProviderEndpointConfig,
    )
    from hcaps.providers.config import (
        ProviderGateConfig as _ProviderGateConfig,
    )
    from hcaps.providers.config import (
        ProviderIngressConfig as _ProviderIngressConfig,
    )
    from hcaps.providers.llama_server_runtime import (  # type: ignore
        LlamaServerConfig as _LlamaServerConfig,
    )
    from hcaps.providers.llama_server_runtime import (
        start_llama_server as _start_llama_server,
    )
    from hcaps.training.config import (  # type: ignore
        ComputeBudgetConfig as _ComputeBudgetConfig,
    )
    from hcaps.training.config import (
        CurriculumConfig as _CurriculumConfig,
    )
    from hcaps.training.config import (
        LossWeights as _LossWeights,
    )
    from hcaps.training.config import (
        TrainingConfig as _TrainingConfig,
    )

    build_corpus = _build_corpus
    compile_axt = _compile_axt
    ExperimentOrchestrator = _ExperimentOrchestrator
    ExperimentSuiteConfig = _ExperimentSuiteConfig
    AxtCompileConfig = _AxtCompileConfig
    CorpusBuildConfig = _CorpusBuildConfig
    ProviderIngressConfig = _ProviderIngressConfig
    ProviderEndpointConfig = _ProviderEndpointConfig
    ProviderCascadeConfig = _ProviderCascadeConfig
    ProviderGateConfig = _ProviderGateConfig
    ProviderCacheConfig = _ProviderCacheConfig
    LlamaServerConfig = _LlamaServerConfig
    start_llama_server = _start_llama_server
    TrainingConfig = _TrainingConfig
    LossWeights = _LossWeights
    CurriculumConfig = _CurriculumConfig
    ComputeBudgetConfig = _ComputeBudgetConfig
    default_arm_catalog = _default_arm_catalog
    HAS_CORE = True
except Exception as import_exc:  # pragma: no cover - UI still useful in monitor-only mode
    CORE_IMPORT_ERROR = str(import_exc)
    HAS_CORE = False

# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------
app = FastAPI(title="Axiom UI", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

templates = Jinja2Templates(directory=str(HERE / "templates"))


# --------------------------------------------------------------------------------------
# Pydantic request / response models (strict, match core where possible)
# --------------------------------------------------------------------------------------
class ProviderEndpointForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_id: str = "primary"
    type: Literal["deterministic", "openai_compatible", "python_callable"] = "openai_compatible"
    provider_family: str = "openai_compatible"
    provider_mode: str = "remote"
    model: str = "sonar-pro"
    base_url: str | None = "https://api.perplexity.ai"
    api_key_env: str | None = "PERPLEXITY_API_KEY"
    timeout_seconds: float = 180.0
    max_retries: int = 2


class CascadeGateForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_confidence: float = Field(0.72, ge=0.0, le=1.0)
    high_impact_claim_types: list[str] = Field(
        default_factory=lambda: [
            "causal_claim",
            "measurement_claim",
            "method_claim",
            "scientific_claim",
        ]
    )
    max_escalation_fraction: float = Field(0.25, ge=0.0, le=1.0)
    escalate_on_warnings: bool = True


class DataBudgetForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_sources: int | None = Field(
        12, description="UI-only: sample at most this many source files for controlled volume"
    )
    max_escalation_fraction: float = 0.25
    max_remote_calls_cap: int | None = Field(
        None, description="Soft UI hint; actual is driven by gate + families"
    )
    estimated_families: int | None = None


class CorpusForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_path: str
    cutoff_date: str | None = "2026-01-01"
    max_chunk_chars: int = Field(420, ge=200)
    claim_family_similarity_threshold: float = Field(0.72, ge=0.0, le=1.0)
    strict_temporal_cutoff: bool = True
    pdf_reader_enabled: bool = False


class TrainingForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: Literal["smoke", "pilot", "standard", "large", "custom"] = "smoke"
    model_config_path: str = "configs/model/structured_native_smoke.yaml"
    geometry_config_path: str = "configs/geometry/geometry_learned_smoke.yaml"
    # Allow overriding a few key loss weights for experimentation (still written to wizard copy)
    text_projection_weight: float = 0.2
    geometry_observables_weight: float = 0.2
    # Large scale / custom overrides for multi-hour extraction + training budgets
    max_steps: int | None = None
    max_records_seen: int | None = None
    max_wall_clock_seconds: float | None = None


class LaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_name: str | None = None
    mode: Literal["remote", "integrated"] = "remote"
    corpus: CorpusForm
    providers: dict[str, Any]  # primary + optional escalation + cascade
    data_budget: DataBudgetForm
    training: TrainingForm
    hf_repo: str | None = None  # for integrated: "org/repo:quant" or leave for local_gguf_path
    local_gguf_path: str | None = None  # absolute path to .gguf for pure local no-HF
    escalation_mode: Literal["none", "deterministic", "remote"] = "none"
    escalation_perplexity: bool = False  # legacy; prefer escalation_mode="none" for local P1


class ERunRequest(BaseModel):
    """Minimal request for a standalone arm-E (structured-native + learned geometry) run.

    Geometry wiring is fixed by the pioneer config and the trainer guard; this surface
    only exposes the knobs that matter for an E run (data, length, seed, precision, output).
    """

    model_config = ConfigDict(extra="forbid")
    input_axt_path: str
    run_name: str | None = None
    max_steps: int = 7500
    seed: int = 13
    precision: Literal["fp32", "bf16"] = "fp32"
    output_dir: str = "runs"
    learning_rate: float | None = None
    micro_batch_size: int | None = None


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_name: str | None = None
    stage: str = (
        "idle"  # idle | sampling | corpus | axt | training | scoring | verdict | done | error
    )
    running: bool = False
    logs_tail: list[str] = Field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------------------
# In-memory job + event bus (simple, self-contained)
# --------------------------------------------------------------------------------------
class EventBus:
    """Very small pub/sub for SSE + live updates. No external deps."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    async def publish(self, event: str, data: dict[str, Any]) -> None:
        payload = json.dumps({"event": event, "data": data}, ensure_ascii=False)
        async with self._lock:
            dead: list[asyncio.Queue[str]] = []
            for q in list(self._subscribers):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                self._subscribers.discard(q)


bus = EventBus()

# Captured at startup so worker threads can publish onto the main event loop.
_MAIN_LOOP: asyncio.AbstractEventLoop | None = None

# Current managed job state
_current_job: dict[str, Any] = {
    "run_name": None,
    "thread": None,
    "stop_event": None,
    "logs": deque(maxlen=400),
    "stage": "idle",
    "run_dir": None,
    "error": None,
}
_job_lock = threading.Lock()


def _publish(event: str, data: dict[str, Any]) -> None:
    """Thread-safe publish: schedule bus.publish on the captured main loop.

    Worker threads have no running event loop, so asyncio.create_task fails there.
    """
    loop = _MAIN_LOOP
    if loop is None or not loop.is_running():
        return
    with suppress(RuntimeError):
        asyncio.run_coroutine_threadsafe(bus.publish(event, data), loop)


def _set_stage(stage: str, run_name: str | None = None) -> None:
    with _job_lock:
        _current_job["stage"] = stage
        if run_name:
            _current_job["run_name"] = run_name
    _publish("stage", {"stage": stage, "run_name": run_name})


def _append_log(line: str) -> None:
    with _job_lock:
        _current_job["logs"].append(line)
    _publish("log", {"line": line})


def _get_job_snapshot() -> JobStatus:
    with _job_lock:
        snap = JobStatus(
            run_name=_current_job.get("run_name"),
            stage=_current_job.get("stage", "idle"),
            running=_current_job.get("thread") is not None,
            logs_tail=list(_current_job.get("logs", []))[-80:],
            error=_current_job.get("error"),
        )

    # Merge with persisted state for long-running / restart resilience (UI-only)
    try:
        if UI_STATE_FILE.exists():
            state = json.loads(UI_STATE_FILE.read_text())
            if not snap.run_name and state.get("run_name"):
                snap = snap.model_copy(
                    update={
                        "run_name": state.get("run_name"),
                        "stage": state.get("stage", snap.stage),
                    }
                )
    except Exception:
        pass
    return snap


def _persist_current_job(run_name: str | None, stage: str, start_time: str | None = None) -> None:
    """Persist so long multi-hour runs survive UI server restart (purely UI control surface)."""
    try:
        if run_name:
            data = {
                "run_name": run_name,
                "stage": stage,
                "started_at": start_time or datetime.utcnow().isoformat() + "Z",
                "is_large_scale": "large" in stage
                or True,  # we don't have the req here, approximate
            }
            UI_STATE_FILE.write_text(json.dumps(data, indent=2))
        elif UI_STATE_FILE.exists():
            UI_STATE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Helper: load real presets (read-only, from project tree)
# --------------------------------------------------------------------------------------
def _safe_read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml  # type: ignore  # noqa: PLC0415

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def _json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON (date/Path) to make responses safe."""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def load_presets() -> dict[str, Any]:
    presets: dict[str, Any] = {
        "corpus": [],
        "models": [],
        "geometry": [],
        "training": [],
        "experiments": [],
    }
    # corpus configs
    for p in sorted((CONFIGS_ROOT / "corpus").rglob("*.yaml")):
        data = _safe_read_yaml(p)
        if data:
            presets["corpus"].append(
                {"path": str(p.relative_to(PROJECT_ROOT)), "data": _json_safe(data)}
            )
    for p in sorted((CONFIGS_ROOT / "model").glob("*.yaml")):
        data = _safe_read_yaml(p)
        if data:
            presets["models"].append(
                {"path": str(p.relative_to(PROJECT_ROOT)), "data": _json_safe(data)}
            )
    for p in sorted((CONFIGS_ROOT / "geometry").glob("*.yaml")):
        data = _safe_read_yaml(p)
        if data:
            presets["geometry"].append(
                {"path": str(p.relative_to(PROJECT_ROOT)), "data": _json_safe(data)}
            )
    for p in sorted((CONFIGS_ROOT / "training").glob("*.yaml")):
        data = _safe_read_yaml(p)
        if data:
            presets["training"].append(
                {"path": str(p.relative_to(PROJECT_ROOT)), "data": _json_safe(data)}
            )
    for p in sorted((CONFIGS_ROOT / "experiments").glob("*.yaml")):
        data = _safe_read_yaml(p)
        if data:
            presets["experiments"].append(
                {"path": str(p.relative_to(PROJECT_ROOT)), "data": _json_safe(data)}
            )
    return presets


def list_runs(limit: int = 30) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for d in sorted(RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        manifest = d / "run_manifest.json"
        corpus_manifest = d / "corpus" / "corpus_manifest.json"
        entry: dict[str, Any] = {"name": d.name, "path": str(d), "created": d.stat().st_mtime}
        if manifest.exists():
            try:
                m = json.loads(manifest.read_text())
                entry["arms"] = len(m.get("arms", {}))
                entry["has_verdict"] = (d / "verdict" / "report.md").exists()
            except Exception:
                pass
        if corpus_manifest.exists():
            try:
                cm = json.loads(corpus_manifest.read_text())
                entry["sources"] = cm.get("source_count")
                entry["capsules"] = cm.get("capsule_count")
            except Exception:
                pass
        runs.append(entry)
        if len(runs) >= limit:
            break
    return runs


# --------------------------------------------------------------------------------------
# Data volume control: source sampling (UI-only, produces reproducible subset)
# --------------------------------------------------------------------------------------
def _sample_sources(src_dir: Path, max_n: int, run_name: str) -> Path:
    """Copy at most max_n files (deterministic order) into a UI-owned temp dir.
    This is the primary "amount of data" lever exposed to the user for quick experiments.
    The generated subset dir is recorded in the run's wizard_configs for repro.
    """
    if max_n is None or max_n <= 0:
        return src_dir
    target = UI_DATA_ROOT / run_name / "sources"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src_dir.iterdir() if p.is_file()])[:max_n]
    for f in files:
        shutil.copy2(f, target / f.name)
    # Also copy any sidecars the readers might expect (very small)
    for f in sorted(src_dir.iterdir()):
        if (
            f.is_file()
            and f.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml", ".md"}
            and f not in files
            and len(files) < max_n + 3
        ):
            shutil.copy2(f, target / f.name)  # tiny sidecars for readers
    _append_log(f"[ui] Sampled {len(files)} source files -> {target}")
    return target


# --------------------------------------------------------------------------------------
# Config synthesis (pure, no side effects on core)
# --------------------------------------------------------------------------------------
def _build_provider_config(req: LaunchRequest) -> Any:
    """Construct ProviderIngressConfig (or raw dict fallback) from the form.
    Honors cascade vs simple remote. Keys are never persisted.
    """
    p = req.providers or {}
    prim = p.get("primary", {}) or {}
    primary = (
        ProviderEndpointConfig(
            provider_id=prim.get("provider_id", "ui_primary"),
            type=prim.get("type", "openai_compatible"),
            provider_family=prim.get("provider_family", "perplexity_sonar"),
            provider_mode=prim.get("provider_mode", "remote"),
            model=prim.get("model", "sonar-pro"),
            base_url=prim.get("base_url", "https://api.perplexity.ai"),
            api_key_env=prim.get("api_key_env", "PERPLEXITY_API_KEY"),
            timeout_seconds=float(prim.get("timeout_seconds", 600)),
        )
        if HAS_CORE and ProviderEndpointConfig
        else prim
    )

    escalation = None
    # Support new escalation_mode from UI for "integrated P1 without remote LLM"
    esc_mode = getattr(req, "escalation_mode", None) or (
        p.get("escalation_mode") if isinstance(p, dict) else None
    )
    if esc_mode == "remote" or (getattr(req, "escalation_perplexity", False) and not esc_mode):
        # keep legacy remote
        if p.get("escalation"):
            esc = p["escalation"] or {}
            escalation = (
                ProviderEndpointConfig(**esc) if HAS_CORE and ProviderEndpointConfig else esc
            )
        else:
            # fallback to hardcoded remote if requested
            escalation = (
                ProviderEndpointConfig(
                    provider_id="perplexity_sonar_escalation",
                    type="openai_compatible",
                    provider_family="perplexity_sonar",
                    provider_mode="remote",
                    model="sonar-pro",
                    base_url="https://api.perplexity.ai",
                    api_key_env="PERPLEXITY_API_KEY",
                )
                if HAS_CORE and ProviderEndpointConfig
                else None
            )
    elif esc_mode == "deterministic":
        escalation = (
            ProviderEndpointConfig(
                provider_id="deterministic_escalation",
                type="deterministic",
                provider_family="deterministic",
                provider_mode="deterministic",
                model="deterministic-bootstrap-v0.2",
            )
            if HAS_CORE and ProviderEndpointConfig
            else {"type": "deterministic"}
        )
    # else "none" or default for pure local: escalation stays None
    elif p.get("escalation"):
        esc = p["escalation"] or {}
        escalation = ProviderEndpointConfig(**esc) if HAS_CORE and ProviderEndpointConfig else esc

    cascade_dict = p.get("cascade", {})
    gate = cascade_dict.get("gate", {})
    cascade = (
        ProviderCascadeConfig(
            enabled=cascade_dict.get("enabled", bool(escalation)),
            gate=ProviderGateConfig(
                min_confidence=gate.get("min_confidence", 0.72),
                high_impact_claim_types=gate.get(
                    "high_impact_claim_types", ["causal_claim", "measurement_claim"]
                ),
                max_escalation_fraction=getattr(req.data_budget, "max_escalation_fraction", 0.25)
                if getattr(req, "data_budget", None)
                else 0.25,
                escalate_on_warnings=gate.get("escalate_on_warnings", True),
            ),
            merge_policy=cascade_dict.get(
                "merge_policy", "schema_grounded_confidence_priority_v0.1"
            ),
        )
        if HAS_CORE
        else {
            "enabled": cascade_dict.get("enabled", bool(escalation)),
            "gate": {
                "min_confidence": gate.get("min_confidence", 0.72),
                "high_impact_claim_types": gate.get("high_impact_claim_types", []),
                "max_escalation_fraction": getattr(req.data_budget, "max_escalation_fraction", 0.25)
                if getattr(req, "data_budget", None)
                else 0.25,
                "escalate_on_warnings": gate.get("escalate_on_warnings", True),
            },
            "merge_policy": cascade_dict.get(
                "merge_policy", "schema_grounded_confidence_priority_v0.1"
            ),
        }
    )

    cache = (
        ProviderCacheConfig(
            root_path=Path(p.get("cache", {}).get("root_path", ".cache/axiom/providers")).resolve(),
            mode=p.get("cache", {}).get("mode", "live"),
        )
        if HAS_CORE
        else p.get("cache", {"mode": "live"})
    )

    if HAS_CORE and ProviderIngressConfig:
        return ProviderIngressConfig(
            primary=primary, escalation=escalation, cascade=cascade, cache=cache
        )
    return {"primary": primary, "escalation": escalation, "cascade": cascade, "cache": cache}


def _build_corpus_config(req: LaunchRequest, sampled_input: Path) -> Any:
    cutoff = None
    if req.corpus.cutoff_date:
        try:
            cutoff = date.fromisoformat(req.corpus.cutoff_date)
        except ValueError:
            cutoff = None
    if HAS_CORE and CorpusBuildConfig:
        return CorpusBuildConfig(
            corpus_id=f"ui.{req.run_name or 'adhoc'}.v0",
            dataset_name=req.run_name or "ui-run",
            input_path=sampled_input,
            output_dir=PROJECT_ROOT / "runs" / (req.run_name or "ui-run") / "corpus",
            cutoff_date=cutoff,
            max_chunk_chars=req.corpus.max_chunk_chars,
            claim_family_similarity_threshold=req.corpus.claim_family_similarity_threshold,
            strict_temporal_cutoff=req.corpus.strict_temporal_cutoff,
            pdf_reader_enabled=req.corpus.pdf_reader_enabled,
            providers=_build_provider_config(req),
        )
    # Fallback raw dict for subprocess path
    return {
        "corpus_id": f"ui.{req.run_name or 'adhoc'}.v0",
        "input_path": str(sampled_input),
        "output_dir": str(PROJECT_ROOT / "runs" / (req.run_name or "ui-run") / "corpus"),
        "cutoff_date": req.corpus.cutoff_date,
        "max_chunk_chars": req.corpus.max_chunk_chars,
        "claim_family_similarity_threshold": req.corpus.claim_family_similarity_threshold,
        "providers": _build_provider_config(req),
    }


def _build_training_template(req: LaunchRequest) -> Any:
    profile = req.training.profile
    profiles = {
        "smoke": {"max_steps": 2, "max_records_seen": 24, "max_wall_clock_seconds": 300.0},
        "pilot": {"max_steps": 20, "max_records_seen": 256, "max_wall_clock_seconds": 1800.0},
        "standard": {"max_steps": 100, "max_records_seen": 2048, "max_wall_clock_seconds": 7200.0},
        "large": {
            "max_steps": 500,
            "max_records_seen": 10000,
            "max_wall_clock_seconds": 28800.0,
        },  # ~8 hours
        "custom": {
            "max_steps": 1000,
            "max_records_seen": 50000,
            "max_wall_clock_seconds": 86400.0,
        },  # 24h example
    }
    s = profiles.get(profile, profiles["standard"])

    # Support large scale / custom overrides for multi-hour runs (extraction + training)
    if req.training.max_steps is not None:
        s = dict(s)  # copy
        s["max_steps"] = req.training.max_steps
    if req.training.max_records_seen is not None:
        s = dict(s)
        s["max_records_seen"] = req.training.max_records_seen
    if req.training.max_wall_clock_seconds is not None:
        s = dict(s)
        s["max_wall_clock_seconds"] = req.training.max_wall_clock_seconds

    if not HAS_CORE or TrainingConfig is None:
        return {
            "max_steps": s["max_steps"],
            "profile": profile,
            "max_records_seen": s.get("max_records_seen"),
            "max_wall_clock_seconds": s.get("max_wall_clock_seconds"),
        }

    # Full construction. Placeholders for required fields (overwritten by orchestrator).
    base = TrainingConfig(
        run_name="ui-template",
        seed=13,
        input_axt_path=Path("ui-placeholder.axt"),
        output_dir=RUNS_ROOT,
        model_config_path=Path(req.training.model_config_path),
        arm_name="ui-template-arm",
        max_steps=int(s["max_steps"]),
        compute_budget_config=ComputeBudgetConfig(
            max_train_steps=int(s["max_steps"]),
            max_records_seen=s["max_records_seen"],
            max_wall_clock_seconds=s["max_wall_clock_seconds"],
        ),
        structured_loss_weights=LossWeights(
            structured_axc_out=1.0,
            relation_prediction=1.0,
            provenance_recovery=1.0,
            epistemic_proxy=1.0,
            stability_temporal=1.0,
            uncertainty_calibration=0.5,
            geometry_observables=req.training.geometry_observables_weight,
            geometry_regularization=0.01,
            text_projection=req.training.text_projection_weight,
        ),
        text_projection_loss_weight=req.training.text_projection_weight,
        geometry_loss_weight=req.training.geometry_observables_weight,
        curriculum_config=CurriculumConfig(schedule_name="none"),
    )
    return base


def _full_arm_ids() -> list[str]:
    if HAS_CORE and default_arm_catalog:
        return list(
            default_arm_catalog(
                input_axt_path="",
                model_config="",
                source_content_id="ui",
                seed=13,
            ).keys()
        )
    # Safe default set matching current P6 catalog (from memory of code)
    return [
        "A_flat_text",
        "B_structured_text",
        "C_capsule_text",
        "D_structured_native_no_geometry",
        "E_structured_native_geometry",
        "F_structured_native_no_provenance",
        "G_structured_native_no_relations",
        "H_structured_native_context_shuffle",
        "I_structured_native_provider_shuffle",
        "J_popularity_frequency_control",
    ]


def _prepare_run_dirs(run_name: str) -> tuple[Path, Path]:
    run_dir = RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    wizard = run_dir / "wizard_configs"
    wizard.mkdir(exist_ok=True)
    return run_dir, wizard


# --------------------------------------------------------------------------------------
# The actual pipeline execution (core-aware when possible; always writes repro artifacts)
# --------------------------------------------------------------------------------------
def _default_axt_config(run_name: str, package_path: Path, axt_out: Path) -> Any:
    if HAS_CORE and AxtCompileConfig:
        return AxtCompileConfig(
            input_path=package_path,
            output_path=axt_out,
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
    return {
        "input_path": str(package_path),
        "output_path": str(axt_out),
        "allow_all_without_split": True,
    }


def _run_full_remote(req: LaunchRequest, stop_event: threading.Event) -> Path:  # noqa: PLR0915
    """Non-interactive remote cascade full pipeline. Mirrors CLI but callable from UI."""
    run_name = req.run_name or f"ui_remote_{int(time.time())}"
    _set_stage("sampling", run_name)
    src = Path(req.corpus.input_path).expanduser().resolve()
    sampled = _sample_sources(src, req.data_budget.max_sources or 0, run_name)

    run_dir, wizard_dir = _prepare_run_dirs(run_name)
    providers = _build_provider_config(req)

    corpus_out = run_dir / "corpus"
    corpus_cfg = _build_corpus_config(req, sampled)
    if HAS_CORE and isinstance(corpus_cfg, CorpusBuildConfig):
        corpus_cfg.to_yaml(wizard_dir / "corpus.yaml")
    else:
        (wizard_dir / "corpus.yaml").write_text(json.dumps(corpus_cfg, indent=2), encoding="utf-8")

    if HAS_CORE and providers is not None:
        # write providers snapshot (best effort, never secrets)
        with suppress(Exception):
            (wizard_dir / "providers.json").write_text(
                json.dumps(
                    providers.model_dump(mode="json")
                    if hasattr(providers, "model_dump")
                    else providers,
                    indent=2,
                ),
                encoding="utf-8",
            )

    _set_stage("corpus", run_name)
    _append_log(f"[ui] Starting corpus build for {run_name} (remote/cascade mode)")
    if HAS_CORE and build_corpus and isinstance(corpus_cfg, CorpusBuildConfig):
        corpus_result = build_corpus(corpus_cfg)
    else:
        # Fallback: shell out to public CLI (still no core modification)
        cfg_path = wizard_dir / "corpus.yaml"
        cmd = [str(PROJECT_ROOT / "bin" / "axiom"), "corpus", "build", "--config", str(cfg_path)]
        _append_log(f"[ui] $ {' '.join(cmd)}")
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1800, check=False
        )
        _append_log(proc.stdout or "")
        if proc.returncode != 0:
            _append_log(proc.stderr or "")
            raise RuntimeError("corpus build via CLI failed")
        # locate the package that was just written (best effort)
        pkg = corpus_out / "dataset.axp"
        corpus_result = type(
            "R",
            (),
            {
                "package_path": pkg,
                "manifest": type("M", (), {"source_count": 0, "capsule_count": 0})(),
            },
        )()

    src_c = getattr(corpus_result.manifest, "source_count", "?")
    cap_c = getattr(corpus_result.manifest, "capsule_count", "?")
    _append_log(f"[ui] Corpus complete: sources={src_c} capsules={cap_c}")

    # AXT
    _set_stage("axt", run_name)
    axt_path = run_dir / "axt" / f"{run_name}.axt"
    axt_cfg = _default_axt_config(run_name, corpus_result.package_path, axt_path)
    if HAS_CORE and compile_axt:
        axt_res = compile_axt(axt_cfg, force=True)  # type: ignore[arg-type]
        _append_log(f"[ui] AXT: {axt_res.manifest.record_count} records -> {axt_path}")
    else:
        cmd = [
            str(PROJECT_ROOT / "bin" / "axiom"),
            "axt",
            "compile",
            "--input",
            str(corpus_result.package_path),
            "--output",
            str(axt_path),
            "--force",
        ]
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    # Experiment suite (full arm catalog, smoke-scale, geometry as chosen)
    _set_stage("training", run_name)
    tmpl = _build_training_template(req)
    suite = ExperimentSuiteConfig(
        suite_name=run_name,
        input_axp=None,
        input_axt=axt_path,
        output_dir=RUNS_ROOT,
        arms=_full_arm_ids(),
        seeds=[13],
        budget_profile="standard"
        if req.training.profile in ("large", "custom")
        else req.training.profile,
        trainer_config_template=tmpl,
        model_config_template=Path(req.training.model_config_path),
        geometry_config_template=Path(req.training.geometry_config_path),
        scoring_config=Path("configs/scoring/default_structured_scoring.yaml"),
        verdict_config=Path("configs/verdict/conservative_thresholds.yaml"),
        source_content_id=run_name,
        split_id="all_without_split",
    )
    suite.to_yaml(wizard_dir / "suite.yaml")
    _append_log(f"[ui] Suite written; launching orchestrator with {len(suite.arms)} arms")

    orchestrator = ExperimentOrchestrator(suite)  # type: ignore[operator]
    exp_res = orchestrator.run()
    _set_stage("verdict", run_name)
    _append_log(f"[ui] Pipeline finished -> {exp_res.run_dir}")
    _set_stage("done", run_name)
    return exp_res.run_dir


def _run_integrated(req: LaunchRequest, stop_event: threading.Event) -> Path:  # noqa: PLR0915
    """Integrated P1 (local-first, no remote LLM by default).
    Supports hf_repo (auto download via llama-server --hf-repo) or local_gguf_path.
    escalation_mode controls remote vs pure local / deterministic cascade.
    """
    if not (HAS_CORE and start_llama_server and LlamaServerConfig):
        raise RuntimeError(
            "Integrated needs core llama_server (import failed). Use remote or terminal wizard."
        )

    run_name = req.run_name or f"ui_integrated_{int(time.time())}"
    _set_stage("sampling", run_name)
    src = Path(req.corpus.input_path).expanduser().resolve()
    sampled = _sample_sources(src, req.data_budget.max_sources or 0, run_name)

    run_dir, wizard_dir = _prepare_run_dirs(run_name)

    hf_repo = (req.hf_repo or "").strip() or None
    local_gguf = (req.local_gguf_path or "").strip() or None
    if not hf_repo and not local_gguf:
        raise ValueError(
            "For integrated mode provide hf_repo (e.g. 'Qwen/...-GGUF') or local_gguf_path"
        )

    # Determine escalation for "without a remote llm"
    esc = None
    esc_mode = getattr(req, "escalation_mode", "none") or "none"
    if esc_mode == "remote" or getattr(req, "escalation_perplexity", False):
        # only if user explicitly wants and key present (env)
        if os.environ.get("PERPLEXITY_API_KEY"):
            esc = ProviderEndpointConfig(  # type: ignore[operator]
                provider_id="perplexity_sonar_escalation",
                type="openai_compatible",
                provider_family="perplexity_sonar",
                provider_mode="remote",
                model="sonar-pro",
                base_url="https://api.perplexity.ai",
                api_key_env="PERPLEXITY_API_KEY",
                timeout_seconds=180.0,
                max_retries=2,
                dry_run=False,
            )
    elif esc_mode == "deterministic":
        esc = ProviderEndpointConfig(  # type: ignore[operator]
            provider_id="deterministic_escalation",
            type="deterministic",
            provider_family="deterministic",
            provider_mode="deterministic",
            model="deterministic-bootstrap-v0.2",
        )

    log_path = run_dir / "llama_server" / "llama-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _set_stage("corpus", run_name)
    _append_log("[ui] Starting integrated llama-server (local P1 extraction, no remote required)")

    env_name = "AXIOM_LLAMA_SERVER_API_KEY"
    primary = ProviderEndpointConfig(  # type: ignore[operator]
        provider_id="integrated_llama_server",
        type="openai_compatible",
        provider_family="llama.cpp",
        provider_mode="local",
        model=hf_repo or Path(local_gguf).name if local_gguf else "ui-local-gguf",
        base_url="http://127.0.0.1:0",
        api_key_env=env_name,
        timeout_seconds=120.0,
        max_retries=1,
        dry_run=False,
    )

    cascade_enabled = esc_mode != "none" or True  # even for none, cascade gate can be present
    cascade = ProviderCascadeConfig(  # type: ignore[operator]
        enabled=bool(cascade_enabled),
        gate=ProviderGateConfig(
            min_confidence=0.72,
            high_impact_claim_types=[
                "causal_claim",
                "measurement_claim",
                "method_claim",
                "scientific_claim",
            ],
            max_escalation_fraction=req.data_budget.max_escalation_fraction or 0.0,
            escalate_on_warnings=True,
        ),
        merge_policy="schema_grounded_confidence_priority_v0.1",
    )

    providers = ProviderIngressConfig(  # type: ignore[operator]
        primary=primary,
        escalation=esc,
        cascade=cascade,
        cache=ProviderCacheConfig(root_path=Path(".cache/axiom/providers").resolve(), mode="live"),
    )

    corpus_cfg = CorpusBuildConfig(  # type: ignore[operator]
        corpus_id=f"ui.{run_name}.v0",
        dataset_name=run_name,
        input_path=sampled,
        output_dir=run_dir / "corpus",
        cutoff_date=date.fromisoformat(req.corpus.cutoff_date) if req.corpus.cutoff_date else None,
        max_chunk_chars=req.corpus.max_chunk_chars,
        claim_family_similarity_threshold=req.corpus.claim_family_similarity_threshold,
        providers=providers,
    )
    corpus_cfg.to_yaml(wizard_dir / "corpus.yaml")
    (wizard_dir / "providers.json").write_text(
        json.dumps(
            providers.model_dump(mode="json") if hasattr(providers, "model_dump") else providers,
            indent=2,
        ),
        encoding="utf-8",
    )

    llama_kwargs: dict[str, Any] = {
        "log_path": log_path,
        "status_callback": lambda m: _append_log(f"[llama] {m}"),
    }
    if local_gguf:
        llama_kwargs["local_gguf_path"] = local_gguf
    else:
        llama_kwargs["hf_repo"] = hf_repo

    with start_llama_server(  # type: ignore[operator]
        LlamaServerConfig(**llama_kwargs)
    ) as server:
        primary.base_url = server.base_url
        primary.model = server.model_name
        os.environ[env_name] = server.api_key
        _append_log(f"[ui] llama-server ready at {server.base_url} (model={server.model_name})")
        corpus_result = build_corpus(corpus_cfg)  # type: ignore[operator]
        _append_log(
            f"[ui] Corpus (integrated local P1) complete: {corpus_result.manifest.source_count} sources"  # noqa: E501
        )

    _append_log("[ui] Integrated llama-server stopped")

    # continue exactly as remote path
    _set_stage("axt", run_name)
    axt_path = run_dir / "axt" / f"{run_name}.axt"
    axt_cfg = _default_axt_config(run_name, corpus_result.package_path, axt_path)
    axt_res = compile_axt(axt_cfg, force=True)  # type: ignore[operator]
    _append_log(f"[ui] AXT compiled: {axt_res.manifest.record_count} records")

    _set_stage("training", run_name)
    tmpl = _build_training_template(req)
    suite = ExperimentSuiteConfig(
        suite_name=run_name,
        input_axp=None,
        input_axt=axt_path,
        output_dir=RUNS_ROOT,
        arms=_full_arm_ids(),
        seeds=[13],
        budget_profile="standard"
        if req.training.profile in ("large", "custom")
        else req.training.profile,
        trainer_config_template=tmpl,
        model_config_template=Path(req.training.model_config_path),
        geometry_config_template=Path(req.training.geometry_config_path),
        scoring_config=Path("configs/scoring/default_structured_scoring.yaml"),
        verdict_config=Path("configs/verdict/conservative_thresholds.yaml"),
        source_content_id=run_name,
        split_id="all_without_split",
    )
    suite.to_yaml(wizard_dir / "suite.yaml")
    orchestrator = ExperimentOrchestrator(suite)  # type: ignore[operator]
    exp_res = orchestrator.run()
    _set_stage("done", run_name)
    return exp_res.run_dir


def _pipeline_worker(req: LaunchRequest, stop_event: threading.Event) -> None:
    run_name: str | None = None
    try:
        with _job_lock:
            _current_job["error"] = None
            _current_job["run_dir"] = None
        run_name = req.run_name or f"ui_{req.mode}_{int(time.time())}"
        req.run_name = run_name  # normalize

        is_large = req.training.profile in ("large", "custom")
        _persist_current_job(run_name, "large-scale-running" if is_large else "running")

        if req.mode == "integrated":
            final = _run_integrated(req, stop_event)
        else:
            final = _run_full_remote(req, stop_event)

        with _job_lock:
            _current_job["run_dir"] = str(final)
        _append_log(f"[ui] Run complete: {final}")
        _persist_current_job(run_name, "done-large" if is_large else "done")
        asyncio.run(bus.publish("done", {"run_dir": str(final), "run_name": run_name}))
    except Exception as e:
        with _job_lock:
            _current_job["error"] = str(e)
        _append_log(f"[ui][ERROR] {e}")
        _persist_current_job(run_name, "error")
        asyncio.run(bus.publish("error", {"message": str(e)}))
    finally:
        with _job_lock:
            _current_job["thread"] = None
            _current_job["stop_event"] = None
        # leave the persisted "done" or "error" state so attach still works after restart


@app.post("/api/launch")
async def launch(req: LaunchRequest) -> JSONResponse:
    with _job_lock:
        if _current_job.get("thread"):
            raise HTTPException(
                409, "A job is already running. Stop it first or attach to monitor an external run."
            )
        stop_ev = threading.Event()
        req.run_name = (
            req.run_name or f"ui_{req.mode}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )
        t = threading.Thread(target=_pipeline_worker, args=(req, stop_ev), daemon=True)
        _current_job["thread"] = t
        _current_job["stop_event"] = stop_ev
        _current_job["logs"].clear()
        _current_job["stage"] = "starting"
        _current_job["run_name"] = req.run_name
        t.start()
    start_iso = datetime.utcnow().isoformat() + "Z"
    _persist_current_job(
        req.run_name,
        "starting-large-scale" if req.training.profile in ("large", "custom") else "starting",
        start_iso,
    )
    _append_log(
        f"[ui] Launched {req.mode} pipeline as {req.run_name} "
        "(large-scale budgets for multi-hour extraction + training)"
    )
    return JSONResponse({"ok": True, "run_name": req.run_name})


@app.post("/api/stop")
async def stop_job() -> JSONResponse:
    with _job_lock:
        t = _current_job.get("thread")
        ev = _current_job.get("stop_event")
        if ev:
            ev.set()
        if t and t.is_alive():
            # Best effort; real stop is cooperative inside the worker
            _append_log("[ui] Stop requested (SIGTERM to children where possible)")
        # Clear persisted state so next load doesn't think a dead job is still current
        _persist_current_job(None, "stopped")
    # Also try to be nice to stray llama-server if any (user can kill manually)
    return JSONResponse({"ok": True})


@app.post("/api/attach")
async def attach_run(payload: dict[str, Any]) -> JSONResponse:
    run_name = payload.get("run_name")
    if not run_name:
        raise HTTPException(400, "run_name required")
    rd = RUNS_ROOT / run_name
    if not rd.exists():
        raise HTTPException(404, "run not found")
    with _job_lock:
        _current_job["run_name"] = run_name
        _current_job["run_dir"] = str(rd)
        _current_job["stage"] = "monitoring"
    _append_log(f"[ui] Attached monitor to {run_name}")
    asyncio.create_task(bus.publish("stage", {"stage": "monitoring", "run_name": run_name}))  # noqa: RUF006
    return JSONResponse({"ok": True, "run_dir": str(rd)})


# --------------------------------------------------------------------------------------
# Standalone arm-E run (structured-native + learned geometry) — consolidated surface
# --------------------------------------------------------------------------------------
_PIONEER_E_CONFIG = (
    PROJECT_ROOT / "configs" / "training" / "pioneer_e_structured_native_geometry.yaml"
)


def _build_e_run_config(req: ERunRequest, run_name: str) -> Path:
    """Derive the E-run YAML from the pioneer config, applying only E-run knobs."""
    if not _PIONEER_E_CONFIG.exists():
        raise RuntimeError(f"E-run config not found: {_PIONEER_E_CONFIG}")
    cfg = yaml.safe_load(_PIONEER_E_CONFIG.read_text(encoding="utf-8")) or {}
    # Apply only the E-run knobs; geometry wiring stays exactly as the pioneer config.
    cfg["run_name"] = run_name
    cfg["input_axt_path"] = req.input_axt_path
    cfg["max_steps"] = int(req.max_steps)
    cfg["seed"] = int(req.seed)
    cfg["precision"] = req.precision
    cfg["output_dir"] = req.output_dir
    budget = dict(cfg.get("compute_budget_config") or {})
    budget["max_train_steps"] = int(req.max_steps)
    cfg["compute_budget_config"] = budget
    if req.learning_rate is not None:
        cfg["learning_rate"] = float(req.learning_rate)
    if req.micro_batch_size is not None:
        mb = max(1, int(req.micro_batch_size))
        cfg["micro_batch_size"] = mb
        cfg["global_batch_size"] = mb
        cfg["gradient_accumulation_steps"] = 1
    run_dir = RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    derived = run_dir / "e_run.yaml"
    derived.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return derived


def _run_e_training(req: ERunRequest, stop_event: threading.Event) -> None:
    run_name = req.run_name or f"e_run_{int(time.time())}"
    try:
        with _job_lock:
            _current_job["error"] = None
            _current_job["run_dir"] = None
        _persist_current_job(run_name, "running")
        _set_stage("preparing", run_name)
        derived = _build_e_run_config(req, run_name)
        run_dir = derived.parent
        with _job_lock:
            _current_job["run_dir"] = str(run_dir)
        _append_log("[ui] Arm E — structured-native + learned geometry")
        _append_log(
            f"[ui] data: {req.input_axt_path}  steps: {req.max_steps}  precision: {req.precision}"
        )
        _append_log(f"[ui] derived config: {derived}")
        _set_stage("training", run_name)
        cmd = [str(PROJECT_ROOT / "bin" / "axiom"), "train", "run", str(derived)]
        _append_log(f"[ui] $ {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout is not None:
            for raw in proc.stdout:
                if stop_event.is_set():
                    proc.terminate()
                    _append_log("[ui] stop requested — terminating E run")
                    break
                line = raw.rstrip()
                if line:
                    _append_log(line)
        rc = proc.wait()
        if stop_event.is_set():
            _persist_current_job(run_name, "stopped")
            _set_stage("stopped", run_name)
        elif rc == 0:
            arm_dir = run_dir / "arms" / "E_structured_native_geometry"
            _append_log(f"[ui] E run complete — checkpoints: {arm_dir}/checkpoints/")
            _persist_current_job(run_name, "done")
            _set_stage("done", run_name)
            _publish("done", {"run_dir": str(run_dir), "run_name": run_name})
        else:
            raise RuntimeError(
                f"axiom train run exited with code {rc}. If the geometry guard fired, the model "
                "config is not in geometry_provider_injected mode (see log above)."
            )
    except Exception as e:
        with _job_lock:
            _current_job["error"] = str(e)
        _append_log(f"[ui][ERROR] {e}")
        _persist_current_job(run_name, "error")
        _publish("error", {"message": str(e)})
    finally:
        with _job_lock:
            _current_job["thread"] = None
            _current_job["stop_event"] = None


@app.post("/api/launch_e")
async def launch_e(req: ERunRequest) -> JSONResponse:
    if not req.input_axt_path or not req.input_axt_path.strip():
        raise HTTPException(400, "input_axt_path (compiled .axt bundle) is required for an E run.")
    with _job_lock:
        if _current_job.get("thread"):
            raise HTTPException(409, "A job is already running. Stop it first.")
        stop_ev = threading.Event()
        req.run_name = req.run_name or f"e_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        t = threading.Thread(target=_run_e_training, args=(req, stop_ev), daemon=True)
        _current_job["thread"] = t
        _current_job["stop_event"] = stop_ev
        _current_job["logs"].clear()
        _current_job["stage"] = "starting"
        _current_job["run_name"] = req.run_name
        t.start()
    _persist_current_job(req.run_name, "starting", datetime.utcnow().isoformat() + "Z")
    _append_log(f"[ui] Launched arm-E run as {req.run_name}")
    return JSONResponse({"ok": True, "run_name": req.run_name})


# --------------------------------------------------------------------------------------
# Config preparation (pure, returns everything needed for repro or manual launch)
# --------------------------------------------------------------------------------------
@app.post("/api/config/prepare")
async def prepare_config(req: LaunchRequest) -> JSONResponse:
    run_name = req.run_name or f"ui_preview_{int(time.time())}"
    src = Path(req.corpus.input_path).expanduser().resolve()
    sampled = (
        _sample_sources(src, req.data_budget.max_sources or 0, run_name)
        if req.data_budget.max_sources
        else src
    )

    corpus_cfg = _build_corpus_config(req, sampled)
    prov_cfg = _build_provider_config(req)
    training_tmpl = _build_training_template(req)

    # Also emit a minimal launch script the user can save & run later
    script = f"""#!/bin/bash
set -e
RUN="{run_name}"
# 1. (optional) sample was prepared by UI at {sampled}
# 2. Corpus
./bin/axiom corpus build --config runs/$RUN/wizard_configs/corpus.yaml || true
# 3. The rest follows the same AXT + experiment steps the UI performed.
echo "See runs/$RUN/wizard_configs/ for exact snapshots and use axiom experiment run on the suite."
"""
    return JSONResponse(
        {
            "run_name": run_name,
            "corpus_yaml": (
                corpus_cfg.model_dump(mode="json")
                if hasattr(corpus_cfg, "model_dump")
                else corpus_cfg
            ),
            "providers": (
                prov_cfg.model_dump(mode="json") if hasattr(prov_cfg, "model_dump") else prov_cfg
            ),
            "training_template": (
                training_tmpl.model_dump(mode="json")
                if hasattr(training_tmpl, "model_dump")
                else training_tmpl
            ),
            "axt_hints": {
                "include_geometry_slots": True,
                "include_negative_samples": True,
                "strict_temporal_masks": True,
            },
            "launch_script": script,
            "data_budget": req.data_budget.model_dump(),
            "note": "Keys are never written. Use cache_only in providers for offline replay.",
        }
    )


# --------------------------------------------------------------------------------------
# Read-only discovery endpoints
# --------------------------------------------------------------------------------------
@app.get("/api/presets")
async def get_presets() -> JSONResponse:
    return JSONResponse(load_presets())


@app.get("/api/models")
async def api_local_models() -> JSONResponse:
    """Automatic recognition of already installed/cached GGUF models + suggestions.
    Powers the WebUI extraction model picker for integrated P1 (local, auto-download supported).
    """
    if not HAS_CORE:
        return JSONResponse(
            {
                "cached": [],
                "suggestions": ["Qwen/Qwen2.5-3B-Instruct-GGUF"],
                "note": "core not loaded",
            }
        )
    try:
        from hcaps.providers.llama_server_runtime import (  # noqa: PLC0415
            get_suggested_gguf_models,
            list_cached_gguf_models,
        )

        cached = list_cached_gguf_models()
        sugg = get_suggested_gguf_models()
        llama_ok = shutil.which("llama-server") is not None
        return JSONResponse(
            {
                "cached": cached,
                "suggestions": sugg,
                "llama_server_on_path": bool(llama_ok),
                "note": "HF models auto-download on launch via llama-server --hf-repo + HF cache. Local .gguf used directly with -m.",  # noqa: E501
            }
        )
    except Exception as e:
        return JSONResponse(
            {"cached": [], "suggestions": ["Qwen/Qwen2.5-3B-Instruct-GGUF"], "error": str(e)}
        )


@app.post("/api/harvest")
async def api_harvest(payload: dict | None = Body(default=None)) -> JSONResponse:  # noqa: PLR0915, PLR0912, B008
    """Internet data collection for P1 sources using Firecrawl, Brave, or simple fetch.
    Saves .md + sidecar metadata so ingest + corpus builder can use them (source_url etc preserved).
    Returns dir usable as corpus.input_path.
    """
    payload = payload or {}
    tool = (payload.get("tool") or "brave").lower()
    query = (payload.get("query") or "").strip()
    count = int(payload.get("count") or 5)
    urls = payload.get("urls") or ([query] if query.startswith("http") else [])
    ts = int(time.time())
    out_dir = UI_DATA_ROOT / "web_harvest" / f"harvest_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    err = None

    def _safe(n: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in n)[:40]

    if tool in ("brave", "search"):
        key = os.environ.get("BRAVE_API_KEY") or payload.get("api_key")
        if not key:
            err = "BRAVE_API_KEY not set (export or pass in payload)"
        else:
            try:
                import json as _json  # noqa: PLC0415
                import urllib.parse  # noqa: PLC0415
                import urllib.request  # noqa: PLC0415

                params = urllib.parse.urlencode({"q": query or " ", "count": min(count, 20)})
                req = urllib.request.Request(
                    f"https://api.search.brave.com/res/v1/web/search?{params}",
                    headers={"Accept": "application/json", "X-Subscription-Token": key},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = _json.loads(resp.read())
                web_results = (data.get("web", {}) or {}).get("results", [])[:count]
                for i, w in enumerate(web_results):
                    title = w.get("title", f"result-{i}")
                    url = w.get("url", "")
                    desc = w.get("description", "")
                    p = out_dir / f"brave_{i:02d}_{_safe(title)}.md"
                    meta = {
                        "title": title,
                        "source_url": url,
                        "domain": (w.get("meta_url") or {}).get("hostname", ""),
                        "provider": "brave",
                        "query": query,
                    }
                    p.write_text(
                        f"---\n{json.dumps(meta, indent=2)}\n---\n\n# {title}\n\n{desc}\n\nSource: {url}\n",  # noqa: E501
                        encoding="utf-8",
                    )
                    (out_dir / f"brave_{i:02d}_{_safe(title)}.json").write_text(
                        json.dumps(meta, indent=2)
                    )
                    items.append({"title": title, "url": url, "saved": str(p)})
            except Exception as ex:
                err = f"brave failed: {ex}"

    if tool in ("firecrawl", "scrape") and not err:
        try:
            from firecrawl import FirecrawlApp  # type: ignore  # noqa: PLC0415

            fc_key = os.environ.get("FIRECRAWL_API_KEY") or payload.get("api_key")
            if not fc_key:
                raise RuntimeError("FIRECRAWL_API_KEY required")
            app = FirecrawlApp(api_key=fc_key)
            targets = urls or ([query] if query.startswith("http") else [])
            for i, t in enumerate(targets[:count]):
                try:
                    res = app.scrape_url(t, params={"formats": ["markdown", "html"]})
                    md = res.get("markdown") or res.get("content") or ""
                    title = (res.get("metadata") or {}).get("title", t.split("/")[-1])
                    p = out_dir / f"firecrawl_{i:02d}_{_safe(title)}.md"
                    meta = {
                        "title": title,
                        "source_url": t,
                        "provider": "firecrawl",
                        "query": query,
                    }
                    p.write_text(
                        f"---\n{json.dumps(meta, indent=2)}\n---\n\n{md}\n", encoding="utf-8"
                    )
                    (out_dir / f"firecrawl_{i:02d}_{_safe(title)}.json").write_text(
                        json.dumps(meta)
                    )
                    items.append({"title": title, "url": t, "saved": str(p)})
                except Exception as ex:
                    items.append({"title": t, "error": str(ex)})
        except ImportError:
            err = "firecrawl-py not installed. Run: uv pip install firecrawl-py"
        except Exception as ex:
            err = f"firecrawl error: {ex}"

    if tool == "fetch" and urls:
        for i, u in enumerate(urls[:count]):
            try:
                import urllib.request  # noqa: PLC0415

                with urllib.request.urlopen(u, timeout=12) as r:
                    raw = r.read().decode("utf-8", "ignore")[:200000]
                title = u.split("/")[-1] or f"page-{i}"
                p = out_dir / f"fetched_{i:02d}_{_safe(title)}.txt"
                meta = {"title": title, "source_url": u, "provider": "fetch", "query": query}
                p.write_text(f"---\n{json.dumps(meta, indent=2)}\n---\n\n{raw}\n")
                (out_dir / f"fetched_{i:02d}_{_safe(title)}.json").write_text(json.dumps(meta))
                items.append({"title": title, "url": u, "saved": str(p)})
            except Exception as ex:
                items.append({"url": u, "error": str(ex)})

    manifest = {"tool": tool, "query": query, "count": len(items), "created": ts, "items": items}
    (out_dir / "harvest_manifest.json").write_text(json.dumps(manifest, indent=2))
    if err and not items:
        return JSONResponse({"error": err, "dir": str(out_dir)})
    return JSONResponse(
        {
            "dir": str(out_dir),
            "count": len(items),
            "items": items,
            "manifest": str(out_dir / "harvest_manifest.json"),
        }
    )


@app.get("/api/runs")
async def get_runs() -> JSONResponse:
    return JSONResponse({"runs": list_runs()})


@app.get("/api/status")
async def status() -> JSONResponse:
    snap = _get_job_snapshot()
    # enrich with FS if monitoring
    if snap.run_name:
        rd = RUNS_ROOT / snap.run_name
        if rd.exists():
            snap = snap.model_copy(update={"run_dir": str(rd)})

    payload = snap.model_dump()

    # Add large-scale / long-run friendly info (elapsed, is_large)
    if payload.get("run_name"):
        try:
            if UI_STATE_FILE.exists():
                st = json.loads(UI_STATE_FILE.read_text())
                if st.get("run_name") == payload["run_name"]:
                    started = st.get("started_at")
                    if started:
                        try:
                            dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                            elapsed = (
                                datetime.utcnow().replace(tzinfo=dt.tzinfo) - dt
                            ).total_seconds()
                            payload["elapsed_seconds"] = int(elapsed)
                            payload["elapsed_human"] = (
                                f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"
                            )
                        except Exception:
                            pass
                    payload["is_large_scale"] = "large" in st.get("stage", "") or st.get(
                        "is_large_scale", False
                    )
        except Exception:
            pass

    return JSONResponse(payload)


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "has_core": HAS_CORE,
            "core_import_error": None if HAS_CORE else globals().get("CORE_IMPORT_ERROR"),
            "project_root": str(PROJECT_ROOT),
        }
    )


# --------------------------------------------------------------------------------------
# Live log / event streaming (SSE)
# --------------------------------------------------------------------------------------
@app.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    async def event_generator():
        q = await bus.subscribe()
        try:
            # initial snapshot
            snap = _get_job_snapshot()
            yield f"data: {json.dumps({'event': 'snapshot', 'data': snap.model_dump()})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield f"data: {msg}\n\n"
                except TimeoutError:
                    # heartbeat
                    yield ": keep-alive\n\n"
        finally:
            await bus.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --------------------------------------------------------------------------------------
# Background poller: push FS-derived snapshots (corpus progress, training curves, verdict)
# even when the job is external or long-running.
# --------------------------------------------------------------------------------------
async def _fs_poller() -> None:  # noqa: PLR0912
    last_mtimes: dict[str, float] = {}
    while True:
        await asyncio.sleep(1.8)
        with _job_lock:
            rn = _current_job.get("run_name")
            rd = _current_job.get("run_dir")
        if not rn:
            continue
        run_dir = Path(rd) if rd else (RUNS_ROOT / rn)
        if not run_dir.exists():
            continue

        # corpus manifest
        cm = run_dir / "corpus" / "corpus_manifest.json"
        if cm.exists() and cm.stat().st_mtime != last_mtimes.get(str(cm), 0):
            last_mtimes[str(cm)] = cm.stat().st_mtime
            try:
                data = json.loads(cm.read_text())
                await bus.publish(
                    "corpus_progress",
                    {
                        "source_count": data.get("source_count"),
                        "capsule_count": data.get("capsule_count"),
                    },
                )
            except Exception:
                pass

        # provider trace count (rough escalation volume)
        trace = run_dir / "corpus" / "extraction_trace.jsonl"
        if trace.exists() and trace.stat().st_mtime != last_mtimes.get(str(trace), 0):
            last_mtimes[str(trace)] = trace.stat().st_mtime
            try:
                lines = trace.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                escalated = sum(
                    1 for ln in lines if "escalation" in ln.lower() or "remote" in ln.lower()
                )
                await bus.publish(
                    "extraction_stats", {"trace_lines": len(lines), "escalated_proxy": escalated}
                )
            except Exception:
                pass

        # latest arm metrics (live training view)
        for arm_dir in (run_dir / "arms").glob("*") if (run_dir / "arms").exists() else []:
            tlog = arm_dir / "train_log.jsonl"
            if tlog.exists() and tlog.stat().st_mtime != last_mtimes.get(str(tlog), 0):
                last_mtimes[str(tlog)] = tlog.stat().st_mtime
                try:
                    lines = [
                        ln
                        for ln in tlog.read_text(encoding="utf-8", errors="ignore")
                        .strip()
                        .splitlines()
                        if ln.strip()
                    ]
                    if lines:
                        last = json.loads(lines[-1])
                        await bus.publish(
                            "train_step",
                            {
                                "arm": arm_dir.name,
                                "step": last.get("step"),
                                "loss": last.get("total_loss"),
                                "raw": last,
                            },
                        )
                except Exception:
                    pass

        # verdict
        vmd = run_dir / "verdict" / "report.md"
        if vmd.exists() and vmd.stat().st_mtime != last_mtimes.get(str(vmd), 0):
            last_mtimes[str(vmd)] = vmd.stat().st_mtime
            await bus.publish("verdict_ready", {"path": str(vmd)})

        # rich analysis snapshot for smart UI (when verdict or full manifest present)
        if (
            vmd.exists() or (run_dir / "run_manifest.json").exists()
        ) and vmd.stat().st_mtime != last_mtimes.get("analysis_" + rn, 0):
            last_mtimes["analysis_" + rn] = (
                vmd.stat().st_mtime
                if vmd.exists()
                else (run_dir / "run_manifest.json").stat().st_mtime
            )
            try:
                analysis = build_run_analysis(rn)
                await bus.publish("analysis_snapshot", analysis)
            except Exception:
                pass


@app.on_event("startup")
async def _startup() -> None:
    global _MAIN_LOOP  # noqa: PLW0603
    _MAIN_LOOP = asyncio.get_running_loop()
    asyncio.create_task(_fs_poller())  # noqa: RUF006


# --------------------------------------------------------------------------------------
# Smart result evaluation & intelligent extraction (UI-only layer)
# Deeply reads existing run artifacts (manifests, logs, verdict, traces, metrics)
# to produce consolidated views, charts data, and rule-based insights.
# Never touches or re-runs any core training/eval code.
# --------------------------------------------------------------------------------------
def _safe_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _safe_jsonl_tail(path: Path, n: int = 50) -> list[dict[str, Any]]:
    try:
        lines = [
            ln
            for ln in path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            if ln.strip()
        ][-n:]
        return [json.loads(ln) for ln in lines]
    except Exception:
        return []


def _parse_run_manifest(run_dir: Path) -> dict[str, Any]:
    m = _safe_json(run_dir / "run_manifest.json") or {}
    arms = m.get("arms", {})
    arm_summaries = {}
    for aid, arm in arms.items():
        mets = arm.get("metrics", {})
        train = mets.get("train", {}) or {}
        val = mets.get("validation", {}) or {}
        arm_summaries[aid] = {
            "status": arm.get("status", "unknown"),
            "steps": mets.get("final_step") or train.get("step"),
            "records_seen": mets.get("records_seen"),
            "parameter_count": mets.get("parameter_count", {}).get("trainable"),
            "train_total_loss": train.get("total_loss"),
            "val_total_loss": val.get("total_loss"),
            "key_losses": {
                k.replace("loss_", ""): v
                for k, v in (train.get("metrics") or {}).items()
                if k.startswith("loss_") and isinstance(v, (int, float))
            },
        }
    return {
        "run_id": m.get("run_id"),
        "created_at": m.get("created_at"),
        "arm_count": len(arms),
        "arms": arm_summaries,
        "has_verdict": (run_dir / "verdict" / "report.md").exists(),
    }


def _parse_verdict(run_dir: Path) -> dict[str, Any]:
    vpath = run_dir / "verdict" / "report.md"
    if not vpath.exists():
        return {"present": False}
    text = vpath.read_text(encoding="utf-8", errors="ignore")
    # Lightweight extraction: overall verdict + branch table
    overall = None
    for line in text.splitlines():
        if "Overall verdict:" in line or "overall verdict" in line.lower():
            overall = (
                line.split("`", 2)[-1].split("`")[0] if "`" in line else line.split(":")[-1].strip()
            )
            break
    branches: list[dict[str, str]] = []
    in_table = False
    for line in text.splitlines():
        if "| Branch |" in line:
            in_table = True
            continue
        if in_table and line.strip().startswith("|") and "Branch" not in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 3:  # noqa: PLR2004
                branches.append({"branch": parts[0], "outcome": parts[1], "reason": parts[2]})
        if in_table and not line.strip().startswith("|"):
            break
    return {"present": True, "overall": overall, "branches": branches, "raw_head": text[:1200]}


def _parse_cascade_stats(run_dir: Path) -> dict[str, Any]:
    trace = run_dir / "corpus" / "extraction_trace.jsonl"
    if not trace.exists():
        return {"present": False}
    recs = _safe_jsonl_tail(trace, 500)
    total = len(recs)
    escalated = sum(
        1
        for r in recs
        if "escalation" in str(r.get("provider_trace", "")).lower()
        or r.get("metadata", {}).get("gate_decision", {}).get("escalate")
    )
    disagreements = sum(1 for r in recs if "disagreement" in str(r).lower())
    return {
        "present": True,
        "total_traces": total,
        "escalated": escalated,
        "escalation_rate": round(escalated / max(1, total), 3),
        "disagreements": disagreements,
    }


def _parse_geometry_impact(arm_summaries: dict[str, Any]) -> dict[str, Any]:
    geo_on = None
    geo_off = None
    for aid, s in arm_summaries.items():
        if "geometry" in aid.lower() and "no_geometry" not in aid.lower():
            geo_on = s
        if "no_geometry" in aid.lower():
            geo_off = s
    if not geo_on or not geo_off:
        return {"available": False}
    # Compare val losses where possible
    delta = None
    try:
        if geo_on.get("val_total_loss") and geo_off.get("val_total_loss"):
            delta = round(geo_on["val_total_loss"] - geo_off["val_total_loss"], 4)
    except Exception:
        pass
    return {
        "available": True,
        "geo_on_arm": geo_on,
        "geo_off_arm": geo_off,
        "val_loss_delta_geo_minus_no": delta,
    }


def build_run_analysis(run_name: str) -> dict[str, Any]:
    """Produce a rich, consolidated analysis dict for the UI from finished artifacts only."""
    rd = RUNS_ROOT / run_name
    if not rd.exists():
        return {"run_name": run_name, "error": "not found"}

    manifest = _parse_run_manifest(rd)
    verdict = _parse_verdict(rd)
    cascade = _parse_cascade_stats(rd)
    geo = _parse_geometry_impact(manifest.get("arms", {}))

    # Simple smart insights (rule based, no claims of superiority)
    insights: list[str] = []
    if cascade.get("present"):
        rate = cascade.get("escalation_rate", 0)
        if rate > 0.15:  # noqa: PLR2004
            insights.append(
                f"Cascade escalated ~{int(rate * 100)}% of traces (volume signal for remote spend)."
            )
        if cascade.get("disagreements", 0) > 5:  # noqa: PLR2004
            insights.append("Multiple provider disagreements; merge policy exercised.")
    if geo.get("available") and geo.get("val_loss_delta_geo_minus_no") is not None:
        d = geo["val_loss_delta_geo_minus_no"]
        insights.append(f"Geometry vs no-geo val loss delta: {d:+.4f} (with fairness controls).")
    if verdict.get("overall"):
        insights.append(
            f"Operator verdict recorded as: {verdict['overall']} (bounded research runtime only)."
        )
    if manifest.get("arm_count", 0) >= 8:  # noqa: PLR2004
        insights.append("Full 10-arm catalog executed — strong controlled comparison surface.")

    # Best arm heuristic for structured objectives (example; UI presents transparently)
    best_structured = None
    lowest_struct = float("inf")
    for aid, s in manifest.get("arms", {}).items():
        kl = s.get("key_losses", {})
        struct = kl.get("structured_axc_out") or kl.get("structured") or 999
        if struct < lowest_struct:
            lowest_struct = struct
            best_structured = aid

    return {
        "run_name": run_name,
        "run_dir": str(rd),
        "manifest": manifest,
        "verdict": verdict,
        "cascade": cascade,
        "geometry": geo,
        "insights": insights,
        "best_structured_proxy": best_structured,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/api/runs/{run_name}/analysis")
async def get_run_analysis(run_name: str) -> JSONResponse:
    """Deep, smart extraction + consolidated view + lightweight insights for a finished run."""
    analysis = build_run_analysis(run_name)
    return JSONResponse(analysis)


@app.get("/api/runs/{run_name}/verdict")
async def get_verdict(run_name: str) -> JSONResponse:
    rd = RUNS_ROOT / run_name
    vpath = rd / "verdict" / "report.md"
    if not vpath.exists():
        raise HTTPException(404, "no verdict")
    return JSONResponse({"text": vpath.read_text(encoding="utf-8", errors="ignore")})


# --------------------------------------------------------------------------------------
# The beautiful single-page app is loaded from ui/templates/index.html at runtime.
# This keeps server.py free of hundreds of long artistic lines so ruff stays happy.
# The template is 100% self-contained (CDNs + inline JS canvas for the fine-art viz).
# --------------------------------------------------------------------------------------
def _load_index_html() -> str:
    # Consolidated arm-E run surface is the default; the original full-pipeline
    # page stays at templates/index.html as a fallback.
    for name in ("e_run.html", "index.html"):
        tmpl = HERE / "templates" / name
        if tmpl.exists():
            return tmpl.read_text(encoding="utf-8", errors="replace")
    return (
        "<!doctype html><title>Axiom UI</title><body><p>Optional UI — template missing.</p></body>"
    )


@app.get("/support.js")
async def support_js() -> Response:
    # Serves the White Plane design runtime so e_run.html (the exact .dc.html) boots.
    p = HERE / "templates" / "support.js"
    if not p.exists():
        raise HTTPException(404, "support.js missing")
    return Response(content=p.read_text(encoding="utf-8"), media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return HTMLResponse(_load_index_html())


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AXIOM_UI_PORT", "7860"))
    print(f"Axiom UI (fine-art optional control surface) on http://127.0.0.1:{port}")
    print("Launch with ./bin/axiom-ui for the recommended uv-isolated invocation.")
    print("This UI never mutates core Axiom code, models, or specs.")
    uvicorn.run(
        "ui.server:app",
        host="127.0.0.1",
        port=port,
        reload=False,
        log_level="info",
    )
