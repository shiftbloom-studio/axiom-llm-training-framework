"""Managed llama-server runtime for local provider-backed data construction."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
BYTES_PER_UNIT = 1024.0


@dataclass(frozen=True)
class LlamaServerConfig:
    """Settings needed to launch a local llama-server OpenAI-compatible endpoint."""

    hf_repo: str | None = None
    local_gguf_path: Path | str | None = None
    host: str = "127.0.0.1"
    port: int | None = None
    api_key: str | None = None
    ctx_size: int = 4096
    parallel: int = 1
    startup_timeout_seconds: float = 7200.0
    startup_status_interval_seconds: float = 30.0
    request_timeout_seconds: float = 180.0
    executable: str = "llama-server"
    log_path: Path | None = None
    status_callback: Callable[[str], None] | None = None


@dataclass(frozen=True)
class LlamaServerHandle:
    """Runtime details for a managed llama-server process."""

    process: subprocess.Popen[bytes]
    config: LlamaServerConfig
    api_key: str
    port: int
    log_path: Path

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.port}/v1"

    @property
    def model_name(self) -> str:
        if self.config.local_gguf_path:
            return Path(self.config.local_gguf_path).name
        return self.config.hf_repo or "local-gguf"


class HuggingFaceGgufValidationError(ValueError):
    """Raised when a Hugging Face repo cannot be used by llama-server --hf-repo."""


@contextmanager
def start_llama_server(config: LlamaServerConfig) -> Iterator[LlamaServerHandle]:
    """Start llama-server and stop it when the context exits.
    Supports either hf_repo (auto-download via --hf-repo) or local_gguf_path (-m local file).
    """

    executable = _resolve_executable(config.executable)
    if config.local_gguf_path:
        p = Path(config.local_gguf_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Local GGUF not found: {p}")
    elif config.hf_repo:
        validate_hf_repo_has_gguf(config.hf_repo)
    else:
        raise ValueError("LlamaServerConfig requires either hf_repo or local_gguf_path")
    port = config.port or find_free_tcp_port(config.host)
    api_key = config.api_key or secrets.token_urlsafe(24)
    log_path = config.log_path or Path(".cache/axiom/llama-server") / f"server-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            _server_command(executable, config, port, api_key, log_path),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    handle = LlamaServerHandle(
        process=process,
        config=config,
        api_key=api_key,
        port=port,
        log_path=log_path,
    )
    try:
        _wait_until_ready(handle)
        yield handle
    finally:
        _stop_process(process)


def find_free_tcp_port(host: str = "127.0.0.1") -> int:
    """Return an available local TCP port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def validate_hf_repo_has_gguf(hf_repo: str, *, timeout_seconds: float = 10.0) -> None:
    """Fail early when a Hugging Face repo has no GGUF files for llama-server."""

    model_id = hf_model_id(hf_repo)
    try:
        payload = _fetch_hf_model_payload(model_id, timeout_seconds=timeout_seconds)
    except HTTPError as exc:
        if exc.code in {HTTP_UNAUTHORIZED, HTTP_FORBIDDEN}:
            return
        if exc.code == HTTP_NOT_FOUND:
            msg = f"Hugging Face repository was not found: {model_id}"
            raise HuggingFaceGgufValidationError(msg) from exc
        return
    except URLError, TimeoutError, OSError, json.JSONDecodeError:
        return

    gguf_files = _gguf_files(payload)
    if gguf_files:
        return
    suggestions = _suggest_gguf_repos(model_id, timeout_seconds=timeout_seconds)
    suggestion_text = (
        " Try a GGUF conversion such as: " + ", ".join(suggestions)
        if suggestions
        else " Search Hugging Face for a repo ending in -GGUF."
    )
    msg = (
        f"Hugging Face repository '{model_id}' has no .gguf files. "
        "Integrated extraction uses llama-server, which requires a GGUF model repo."
        f"{suggestion_text}"
    )
    raise HuggingFaceGgufValidationError(msg)


def hf_model_id(hf_repo: str) -> str:
    """Return the repo id from llama.cpp repo[:quant] syntax."""

    return hf_repo.split(":", maxsplit=1)[0].strip()


def _resolve_executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is not None:
        return resolved
    homebrew = Path("/opt/homebrew/bin") / name
    if homebrew.exists():
        return str(homebrew)
    msg = (
        "llama-server executable was not found. Install llama.cpp or ensure "
        "llama-server is on PATH."
    )
    raise FileNotFoundError(msg)


def _fetch_hf_model_payload(model_id: str, *, timeout_seconds: float) -> dict[str, Any]:
    encoded = quote(model_id, safe="/")
    url = f"https://huggingface.co/api/models/{encoded}?expand[]=siblings"
    request = Request(url, method="GET", headers=_hf_headers())
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        msg = f"Hugging Face API returned non-object payload for {model_id}"
        raise HuggingFaceGgufValidationError(msg)
    return payload


def _fetch_hf_search_payload(query: str, *, timeout_seconds: float) -> list[dict[str, Any]]:
    params = urlencode({"search": query, "limit": "5"})
    request = Request(
        f"https://huggingface.co/api/models?{params}",
        method="GET",
        headers=_hf_headers(),
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _gguf_files(payload: dict[str, Any]) -> list[str]:
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        return []
    files: list[str] = []
    for sibling in siblings:
        if not isinstance(sibling, dict):
            continue
        name = sibling.get("rfilename")
        if isinstance(name, str) and name.casefold().endswith(".gguf"):
            files.append(name)
    return files


def _suggest_gguf_repos(model_id: str, *, timeout_seconds: float) -> list[str]:
    query = f"{model_id.rsplit('/', maxsplit=1)[-1]} GGUF"
    try:
        payload = _fetch_hf_search_payload(query, timeout_seconds=timeout_seconds)
    except HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError:
        return []
    suggestions: list[str] = []
    for item in payload:
        candidate = item.get("modelId")
        if not isinstance(candidate, str):
            continue
        if "gguf" not in candidate.casefold():
            continue
        suggestions.append(candidate)
    return suggestions[:3]


def _server_command(
    executable: str,
    config: LlamaServerConfig,
    port: int,
    api_key: str,
    log_path: Path,
) -> list[str]:
    common = [
        "--host",
        config.host,
        "--port",
        str(port),
        "--ctx-size",
        str(config.ctx_size),
        "--parallel",
        str(config.parallel),
        "--api-key",
        api_key,
        "--timeout",
        str(int(config.request_timeout_seconds)),
        "--no-ui",
        "--log-file",
        str(log_path),
    ]
    if config.local_gguf_path:
        p = Path(config.local_gguf_path)
        alias = p.name
        return [executable, "-m", str(p), "--alias", alias, *common]
    # hf path
    repo = config.hf_repo or ""
    return [executable, "--hf-repo", repo, "--alias", repo, *common]


def _wait_until_ready(handle: LlamaServerHandle) -> None:
    started = time.monotonic()
    deadline = time.monotonic() + handle.config.startup_timeout_seconds
    last_error = "server not ready"
    next_status = started + handle.config.startup_status_interval_seconds
    while time.monotonic() < deadline:
        exit_code = handle.process.poll()
        if exit_code is not None:
            tail = _tail_text(handle.log_path)
            msg = f"llama-server exited during startup with code {exit_code}: {tail}"
            raise RuntimeError(msg)
        try:
            request = Request(
                f"{handle.base_url}/models",
                headers={"Authorization": f"Bearer {handle.api_key}"},
                method="GET",
            )
            with urlopen(request, timeout=2.0) as response:
                payload = json.loads(response.read())
            if isinstance(payload, dict):
                return
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            now = time.monotonic()
            if handle.config.status_callback is not None and now >= next_status:
                elapsed = _format_duration(now - started)
                mid = hf_model_id(handle.config.hf_repo) if handle.config.hf_repo else None
                status = (
                    _hf_cache_status(mid) if mid else f"local GGUF: {handle.config.local_gguf_path}"
                )
                handle.config.status_callback(
                    f"llama-server still starting after {elapsed}; last probe: "
                    f"{last_error}; {status}; log: {handle.log_path}"
                )
                next_status = now + handle.config.startup_status_interval_seconds
            time.sleep(1.0)
    tail = _tail_text(handle.log_path)
    mid = hf_model_id(handle.config.hf_repo) if handle.config.hf_repo else None
    status = _hf_cache_status(mid) if mid else f"local GGUF: {handle.config.local_gguf_path}"
    msg = (
        f"llama-server did not become ready after "
        f"{_format_duration(handle.config.startup_timeout_seconds)}: {last_error}. "
        f"{status}. Log tail: {tail}"
    )
    raise TimeoutError(msg)


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _tail_text(path: Path, *, max_bytes: int = 4000) -> str:
    if not path.exists():
        return "(no log file)"
    data = path.read_bytes()
    return data[-max_bytes:].decode("utf-8", errors="replace").strip()


def _hf_cache_status(model_id: str) -> str:
    cache_dir = _hf_cache_dir(model_id)
    if not cache_dir.exists():
        return f"Hugging Face cache not started for {model_id}"
    in_progress = [path for path in cache_dir.glob("blobs/*.downloadInProgress") if path.is_file()]
    if in_progress:
        size = sum(path.stat().st_size for path in in_progress)
        return f"Hugging Face download in progress: {_format_bytes(size)} cached"
    gguf_files = [path for path in cache_dir.rglob("*.gguf") if path.is_file()]
    if gguf_files:
        size = sum(path.stat().st_size for path in gguf_files)
        return f"Hugging Face GGUF cache present: {_format_bytes(size)}"
    size = sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file())
    return f"Hugging Face cache present: {_format_bytes(size)}"


def _hf_cache_dir(model_id: str) -> Path:
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        root = Path(hub_cache).expanduser()
    else:
        hf_home = os.environ.get("HF_HOME")
        root = (
            Path(hf_home).expanduser() / "hub"
            if hf_home
            else Path.home() / ".cache/huggingface/hub"
        )
    return root / f"models--{model_id.replace('/', '--')}"


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < BYTES_PER_UNIT or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= BYTES_PER_UNIT
    return f"{value:.1f} TiB"


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}h {mins}m {secs}s"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


# --------------------------------------------------------------------------------------
# Discovery for WebUI: automatic recognition of cached/installed GGUF models
# (used by integrated local P1 extraction without remote)
# --------------------------------------------------------------------------------------


def list_cached_gguf_models() -> list[dict[str, Any]]:
    """Scan HF hub cache and common local directories for .gguf files.
    Returns list suitable for UI model picker. Supports 'auto download' awareness.
    """
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    # HF cache (re-uses the cache dir logic)
    try:
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        hub_root = Path(hf_home) / "hub"
        if hub_root.exists():
            for model_dir in sorted(hub_root.glob("models--*")):
                repo = model_dir.name.replace("models--", "").replace("--", "/")
                for gguf in sorted(model_dir.rglob("*.gguf")):
                    p = str(gguf)
                    if p in seen:
                        continue
                    seen.add(p)
                    size = gguf.stat().st_size
                    results.append(
                        {
                            "type": "hf",
                            "repo": repo,
                            "path": p,
                            "size_bytes": size,
                            "display": f"{repo} ({gguf.name}, {_format_bytes(size)})",
                            "cached": True,
                            "hf_repo": f"{repo}:{_infer_quant(gguf.name)}"
                            if ":" not in repo
                            else repo,
                        }
                    )
    except Exception:
        pass

    # Common local GGUF locations (user installed)
    for base in [
        Path.home() / "models",
        Path.home() / "gguf",
        Path.home() / ".cache" / "gguf",
        Path.cwd() / "models",
        Path.cwd() / "gguf",
        Path("/opt/llama/models"),
        Path("/opt/homebrew/var/llama/models"),
    ]:
        try:
            if base.exists():
                for gguf in sorted(base.rglob("*.gguf")):
                    p = str(gguf)
                    if p in seen:
                        continue
                    seen.add(p)
                    size = gguf.stat().st_size
                    results.append(
                        {
                            "type": "local",
                            "repo": None,
                            "path": p,
                            "size_bytes": size,
                            "display": f"local:{gguf.name} ({_format_bytes(size)})",
                            "cached": True,
                            "local_gguf_path": p,
                        }
                    )
        except Exception:
            pass

    return results


def _infer_quant(name: str) -> str:
    n = name.lower()
    for q in ("q4_k_m", "q5_k_m", "q4_0", "q5_0", "q8_0", "q3_k_m", "f16"):
        if q in n:
            return q.upper()
    return "Q4_K_M"  # common default


def get_suggested_gguf_models() -> list[str]:
    """Popular small-to-medium GGUF repos good for local extraction (claim/epistemic tasks)."""
    return [
        "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "TheBloke/phi-2-GGUF",
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    ]
