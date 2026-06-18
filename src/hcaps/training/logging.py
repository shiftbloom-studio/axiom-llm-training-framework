"""Artifact logging helpers for P6 training."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import orjson
import torch

from hcaps.utils.time import utc_now


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"
    )


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("ab") as handle:
        handle.write(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")


def environment_summary() -> dict[str, Any]:
    """Return a safe dependency/hardware summary without secrets."""

    return {
        "created_at": utc_now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }


def git_state_summary(cwd: Path) -> dict[str, Any]:
    """Record commit and dirty-state markers without shelling out to secrets."""

    commit = _git(["git", "rev-parse", "HEAD"], cwd)
    status = _git(["git", "status", "--short"], cwd)
    branch = _git(["git", "branch", "--show-current"], cwd)
    return {
        "commit": commit.strip() if commit else None,
        "branch": branch.strip() if branch else None,
        "dirty": bool(status and status.strip()),
        "status_short": status.splitlines() if status else [],
    }


def _git(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = json.loads(handle.read())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
