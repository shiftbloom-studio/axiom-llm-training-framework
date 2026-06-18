"""Dataset-level diagnostics for falsification arms.

These are readiness and confound diagnostics, not model-quality scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hcaps.falsification.audits import (
    audit_forbidden_truth_labels,
    audit_future_target_exposure,
    audit_temporal_leakage,
)
from hcaps.training_bridge.side_channels import extract_side_channels


def compute_dataset_metrics(
    capsules: list[dict[str, Any]],
    rendered_texts: Sequence[str | dict[str, Any]],
    *,
    tokenizer: Any | None = None,
    include_side_channels: bool = True,
) -> dict[str, int | float]:
    """Compute deterministic dataset-level diagnostics for one arm."""

    rendered_strings = [_rendered_text_value(rendered) for rendered in rendered_texts]
    token_counts = [_token_count(text, tokenizer) for text in rendered_strings]
    relation_count = sum(_relation_count(capsule) for capsule in capsules)
    provenance_source_count = sum(_provenance_count(capsule) for capsule in capsules)
    context_ids = [_context_id(capsule) for capsule in capsules if _context_id(capsule)]
    side_channels = (
        [extract_side_channels(capsule) for capsule in capsules] if include_side_channels else []
    )

    return {
        "record_count": len(capsules),
        "total_rendered_characters": sum(len(text) for text in rendered_strings),
        "total_token_count": sum(token_counts),
        "average_token_count": _average(token_counts),
        "max_token_count": max(token_counts) if token_counts else 0,
        "empty_text_count": sum(1 for text in rendered_strings if not text.strip()),
        "relation_count": relation_count,
        "average_relations_per_capsule": relation_count / len(capsules) if capsules else 0.0,
        "provenance_source_count": provenance_source_count,
        "average_sources_per_capsule": provenance_source_count / len(capsules) if capsules else 0.0,
        "context_count": sum(1 for capsule in capsules if isinstance(capsule.get("context"), dict)),
        "unique_context_count": len(set(context_ids)),
        "unique_claim_family_count": len({_claim_family_id(capsule) for capsule in capsules}),
        "side_channel_dimension": len(side_channels[0]) if side_channels else 0,
        "side_channel_missing_rate": _side_channel_missing_rate(capsules, side_channels),
        "temporal_leakage_count": len(audit_temporal_leakage(capsules)),
        "forbidden_truth_label_count": len(audit_forbidden_truth_labels(capsules)),
        "future_target_exposure_count": len(audit_future_target_exposure(rendered_strings)),
    }


def _rendered_text_value(rendered: str | dict[str, Any]) -> str:
    if isinstance(rendered, dict):
        return str(rendered.get("rendered_text", ""))
    return str(rendered)


def _token_count(text: str, tokenizer: Any | None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text.split())


def _relation_count(capsule: dict[str, Any]) -> int:
    relations = capsule.get("relations", [])
    return len(relations) if isinstance(relations, list) else 0


def _provenance_count(capsule: dict[str, Any]) -> int:
    provenance = capsule.get("provenance", [])
    if isinstance(provenance, list):
        return len(provenance)
    if isinstance(provenance, dict):
        sources = provenance.get("sources", [])
        return len(sources) if isinstance(sources, list) else 0
    return 0


def _context_id(capsule: dict[str, Any]) -> str | None:
    context = capsule.get("context", {})
    if isinstance(context, dict):
        value = context.get("context_id") or context.get("primary_context")
        return str(value) if value else None
    return None


def _claim_family_id(capsule: dict[str, Any]) -> str:
    ids = capsule.get("ids", {})
    claim = capsule.get("claim", {})
    if capsule.get("claim_family_id"):
        return str(capsule["claim_family_id"])
    if isinstance(ids, dict) and ids.get("claim_family_id"):
        return str(ids["claim_family_id"])
    if isinstance(claim, dict) and claim.get("claim_id"):
        return str(claim["claim_id"])
    return str(capsule.get("capsule_id", "unknown_claim_family"))


def _side_channel_missing_rate(
    capsules: list[dict[str, Any]], side_channels: list[list[float]]
) -> float:
    if not capsules:
        return 0.0
    if not side_channels:
        return 1.0
    missing = sum(1 for values in side_channels if not values)
    return missing / len(capsules)


def _average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0
