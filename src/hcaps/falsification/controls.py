"""Deterministic control and ablation transforms for AXC dictionaries."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any


def remove_provenance(capsule: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a capsule with provenance removed."""

    transformed = deepcopy(capsule)
    transformed.pop("provenance", None)
    return transformed


def remove_relations(capsule: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a capsule with relation fields removed."""

    transformed = deepcopy(capsule)
    transformed["relations"] = []
    return transformed


def remove_context(capsule: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a capsule with context removed.

    Removing context makes canonical HoloCapsule validation fail for the
    current schema, so runners must mark this artifact as derived and
    noncanonical.
    """

    transformed = deepcopy(capsule)
    transformed.pop("context", None)
    return transformed


def remove_side_channels_from_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return a dataset item copy without numeric side-channel features."""

    transformed = deepcopy(item)
    transformed.pop("side_channels", None)
    return transformed


def shuffle_contexts(capsules: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Shuffle context assignments deterministically while preserving record count."""

    transformed = [deepcopy(capsule) for capsule in capsules]
    contexts = [deepcopy(capsule.get("context")) for capsule in capsules]
    for capsule, context in zip(transformed, _stable_permutation(contexts, seed), strict=True):
        if context is None:
            capsule.pop("context", None)
        else:
            capsule["context"] = context
    return transformed


def shuffle_provenance(capsules: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Shuffle provenance assignments deterministically while preserving per-record counts."""

    transformed = [deepcopy(capsule) for capsule in capsules]
    provenance_lists = [_provenance_list(capsule) for capsule in capsules]
    counts = [len(records) for records in provenance_lists]
    shuffled_records = _stable_permutation(
        [deepcopy(record) for records in provenance_lists for record in records],
        seed,
    )

    cursor = 0
    for capsule, count in zip(transformed, counts, strict=True):
        capsule["provenance"] = shuffled_records[cursor : cursor + count]
        cursor += count
    return transformed


def build_popularity_frequency_control(capsules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a noncanonical control preserving only simple frequency proxies."""

    family_counts: dict[str, int] = {}
    for capsule in capsules:
        family = _claim_family_id(capsule)
        family_counts[family] = family_counts.get(family, 0) + 1

    controls: list[dict[str, Any]] = []
    for capsule in capsules:
        source_count = len(_provenance_list(capsule))
        family = _claim_family_id(capsule)
        relation_count = len(capsule.get("relations", []))
        controls.append(
            {
                "capsule_id": capsule.get("capsule_id") or capsule.get("ids", {}).get("capsule_id"),
                "claim": deepcopy(capsule.get("claim", {})),
                "surface_forms": deepcopy(capsule.get("surface_forms", {})),
                "provenance": [
                    {"source_count_proxy_index": index + 1} for index in range(source_count)
                ],
                "relations": [],
                "popularity_frequency_control": {
                    "source_count_proxy": source_count,
                    "relation_count_proxy": relation_count,
                    "claim_family_id": family,
                    "claim_family_frequency_proxy": family_counts[family],
                },
                "derived_noncanonical": True,
                "derived_from": (
                    capsule.get("capsule_id") or capsule.get("ids", {}).get("capsule_id")
                ),
            }
        )
    return controls


def build_relation_ablation(capsules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove relation fields and relation-derived side-channel signal."""

    return [remove_relations(capsule) for capsule in capsules]


def _provenance_list(capsule: dict[str, Any]) -> list[dict[str, Any]]:
    provenance = capsule.get("provenance", [])
    if isinstance(provenance, list):
        return [record for record in provenance if isinstance(record, dict)]
    if isinstance(provenance, dict):
        sources = provenance.get("sources", [])
        if isinstance(sources, list):
            return [record for record in sources if isinstance(record, dict)]
    return []


def _claim_family_id(capsule: dict[str, Any]) -> str:
    ids = capsule.get("ids", {})
    claim = capsule.get("claim", {})
    if isinstance(ids, dict) and ids.get("claim_family_id"):
        return str(ids["claim_family_id"])
    if capsule.get("claim_family_id"):
        return str(capsule["claim_family_id"])
    if isinstance(claim, dict) and claim.get("claim_id"):
        return str(claim["claim_id"])
    return str(capsule.get("capsule_id", "unknown_claim_family"))


def _stable_permutation(items: list[Any], seed: int) -> list[Any]:
    if len(items) <= 1:
        return [deepcopy(item) for item in items]

    indexed = list(enumerate(items))
    indexed.sort(key=lambda pair: _permutation_key(pair[0], pair[1], seed))
    if [index for index, _ in indexed] == list(range(len(items))):
        shift = seed % len(items)
        if shift == 0:
            shift = 1
        indexed = indexed[shift:] + indexed[:shift]
    return [deepcopy(item) for _, item in indexed]


def _permutation_key(index: int, item: Any, seed: int) -> str:
    stable_hint = ""
    if isinstance(item, dict):
        stable_hint = str(
            item.get("context_id")
            or item.get("source_id")
            or item.get("provenance_id")
            or item.get("capsule_id")
            or ""
        )
    payload = f"{seed}:{index}:{stable_hint}".encode()
    return hashlib.sha256(payload).hexdigest()
