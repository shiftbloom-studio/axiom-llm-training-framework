"""Deterministic renderers for AXC capsules.

This module provides functions to convert AXC capsules into textual
representations for pretraining. The three functions defined here
serve different purposes:

* :func:`render_flat_text` returns the best available surface form
  without any additional structure. Use this as a baseline similar to
  conventional text-only pretraining.
* :func:`render_structured_text` includes high-level labels such as
  claim, context and sources, giving the model hints about epistemic
  fields without overwhelming detail.
* :func:`render_capsule_text` emits a verbose, labelled version of
  the entire capsule. All top-level fields are prefixed by a label,
  and nested values are flattened where appropriate. Future-only
  target fields are excluded, and no truth labels are written.
"""

from __future__ import annotations

from typing import Any


def _claim_text(capsule: dict[str, Any]) -> str:
    claim = capsule.get("claim", {})
    if not isinstance(claim, dict):
        return ""
    return str(claim.get("canonical_text") or claim.get("canonical") or "").strip()


def _get_surface_text(capsule: dict[str, Any]) -> str:
    """Return the best available surface text for a capsule.

    Prefers summary-style surface forms, falling back to primary surface
    text and then the canonical claim when necessary.
    """
    sf = capsule.get("surface_forms", {})
    if not isinstance(sf, dict):
        return _claim_text(capsule)
    for key in ("normalized_summary", "summary", "primary_text", "canonical"):
        value = sf.get(key)
        if value:
            return str(value)
    return _claim_text(capsule)


def _provenance_records(capsule: dict[str, Any]) -> list[dict[str, Any]]:
    provenance = capsule.get("provenance", [])
    if isinstance(provenance, list):
        return [src for src in provenance if isinstance(src, dict)]
    if isinstance(provenance, dict):
        sources = provenance.get("sources", [])
        if isinstance(sources, list):
            return [src for src in sources if isinstance(src, dict)]
    return []


def _context_summary(capsule: dict[str, Any]) -> list[str]:
    context = capsule.get("context", {})
    if not isinstance(context, dict):
        return []

    parts: list[str] = []
    context_id = context.get("context_id") or context.get("primary_context")
    if context_id:
        parts.append(f"Context: {context_id}")
    domains = context.get("domains")
    if isinstance(domains, list) and domains:
        parts.append("Domains: " + ", ".join(str(domain) for domain in domains))
    cutoff = context.get("temporal_cutoff", {})
    if isinstance(cutoff, dict) and cutoff.get("cutoff_at"):
        parts.append(f"Valid as of: {cutoff['cutoff_at']}")
    return parts


def render_flat_text(capsule: dict[str, Any]) -> str:
    """Return a plain-text representation of the capsule."""
    return _get_surface_text(capsule).strip()


def render_structured_text(capsule: dict[str, Any]) -> str:
    """Return a lightly structured representation of the capsule."""
    parts: list[str] = []
    claim = _claim_text(capsule)
    if claim:
        parts.append(f"Claim: {claim}")
    parts.extend(_context_summary(capsule))
    provenance_titles: list[str] = []
    for src in _provenance_records(capsule):
        title = src.get("source_title") or src.get("title")
        if title:
            provenance_titles.append(str(title))
    if provenance_titles:
        parts.append("Sources: " + "; ".join(provenance_titles))
    return "\n".join(parts)


def render_capsule_text(capsule: dict[str, Any]) -> str:
    """Return a verbose labelled representation of the capsule."""
    lines: list[str] = []
    claim = _claim_text(capsule)
    if claim:
        lines.append(f"Claim: {claim}")

    surface_text = _get_surface_text(capsule).strip()
    if surface_text and surface_text != claim:
        lines.append(f"Surface: {surface_text}")

    ep = capsule.get("epistemic_state", {})
    if isinstance(ep, dict):
        for key, value in ep.items():
            if isinstance(value, dict) and "value" in value:
                lines.append(f"{key}: {value['value']}")
            else:
                lines.append(f"{key}: {value}")

    rels = capsule.get("relations", [])
    if isinstance(rels, list) and rels:
        rel_descriptions = [
            f"{rel.get('relation_type') or rel.get('type', '?')} -> "
            f"{rel.get('target_claim_id') or rel.get('target_claim_family_id', '?')}"
            for rel in rels
            if isinstance(rel, dict)
        ]
        if rel_descriptions:
            lines.append("Relations: " + "; ".join(rel_descriptions))

    provenance_titles: list[str] = []
    for src in _provenance_records(capsule):
        title = src.get("source_title") or src.get("title")
        if title:
            provenance_titles.append(str(title))
    if provenance_titles:
        lines.append("Sources: " + "; ".join(provenance_titles))

    lines.extend(_context_summary(capsule))
    return "\n".join(lines)
