"""Deterministic split utilities for falsification harness runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_split_ids(axp_path: str | Path, split_name: str) -> list[str]:
    """Load capsule IDs for a named split from an AXP package."""

    root = Path(axp_path)
    if not root.exists():
        raise FileNotFoundError(f"AXP package does not exist: {root}")
    split_dir = root / "splits"
    candidates = [
        split_dir / f"{split_name}.json",
        split_dir / f"{split_name}.jsonl",
        split_dir / f"{split_name}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _read_split_file(candidate)
    raise FileNotFoundError(
        f"No split named '{split_name}' found in {split_dir}. "
        "Create a split file or rerun with allow_all_without_split=True."
    )


def apply_split(
    capsules: list[dict[str, Any]],
    split_ids: set[str] | list[str],
) -> list[dict[str, Any]]:
    """Select capsules by ID while preserving input order."""

    wanted = {str(split_id) for split_id in split_ids}
    return [capsule for capsule in capsules if _capsule_id(capsule) in wanted]


def make_temporal_holdout(
    capsules: list[dict[str, Any]], cutoff_date: str | datetime
) -> dict[str, list[dict[str, Any]]]:
    """Split capsules by valid-as-of timestamp."""

    cutoff = _parse_datetime(cutoff_date)
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for capsule in capsules:
        valid_as_of = _valid_as_of(capsule)
        if valid_as_of is None:
            holdout.append(capsule)
        elif valid_as_of <= cutoff:
            train.append(capsule)
        else:
            holdout.append(capsule)
    return {"train": train, "holdout": holdout}


def make_deterministic_random_split(
    capsules: list[dict[str, Any]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic random splits from stable capsule IDs."""

    if not ratios:
        raise ValueError("At least one split ratio is required.")
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    ordered = sorted(capsules, key=_capsule_id)
    ordered.sort(key=lambda capsule: _stable_hash(f"{seed}:{_capsule_id(capsule)}"))

    total = len(ordered)
    normalized = {name: ratio / total_ratio for name, ratio in ratios.items()}
    split_names = list(ratios)
    raw_counts = {name: normalized[name] * total for name in split_names}
    counts = {name: int(raw_counts[name]) for name in split_names}
    remaining = total - sum(counts.values())
    remainders = sorted(
        split_names,
        key=lambda name: (raw_counts[name] - counts[name], name),
        reverse=True,
    )
    for name in remainders[:remaining]:
        counts[name] += 1

    splits: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for name in split_names:
        count = counts[name]
        splits[name] = ordered[cursor : cursor + count]
        cursor += count
    return splits


def _read_split_file(path: Path) -> list[str]:
    if path.suffix == ".txt":
        return [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    if path.suffix == ".jsonl":
        ids: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, str):
                ids.append(payload)
            elif isinstance(payload, dict):
                ids.append(str(payload.get("capsule_id") or payload.get("id")))
        return ids

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(value) for value in payload]
    if isinstance(payload, dict):
        split_ids: object = payload.get("capsule_ids") or payload.get("ids")
        if isinstance(split_ids, list):
            return [str(value) for value in split_ids]
    raise ValueError(f"Unsupported split file shape: {path}")


def _capsule_id(capsule: dict[str, Any]) -> str:
    ids = capsule.get("ids", {})
    if capsule.get("capsule_id"):
        return str(capsule["capsule_id"])
    if isinstance(ids, dict) and ids.get("capsule_id"):
        return str(ids["capsule_id"])
    raise ValueError(f"Capsule is missing a capsule ID: {capsule}")


def _valid_as_of(capsule: dict[str, Any]) -> datetime | None:
    temporal = capsule.get("temporal", {})
    if isinstance(temporal, dict) and temporal.get("valid_as_of"):
        return _parse_datetime(temporal["valid_as_of"])
    context = capsule.get("context", {})
    if isinstance(context, dict):
        cutoff = context.get("temporal_cutoff", {})
        if isinstance(cutoff, dict) and cutoff.get("cutoff_at"):
            return _parse_datetime(cutoff["cutoff_at"])
    return None


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    return datetime.fromisoformat(normalized)


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
