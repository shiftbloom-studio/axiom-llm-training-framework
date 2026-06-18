"""Extraction of side-channel features.

Side-channel features are numeric values derived from the capsule's
epistemic and relational metadata. They serve as auxiliary inputs to
the model but are kept separate from the text to avoid conflating
epistemic status with the language signal. All numeric fields are
converted to floats. Missing values default to zero.
"""

from __future__ import annotations

from typing import Any


def extract_side_channels(capsule: dict[str, Any]) -> list[float]:
    """Extract numeric side-channel features from a capsule.

    Features extracted in order:

    1. ontic/ontology compatibility value
    2. evidential_anchoring value
    3. transformation_pressure value
    4. redundancy value
    5. uncertainty value
    6. relation_count
    7. provenance_source_count
    8. geometry_enabled (1.0 if experimental geometry is present, else 0.0)
    """
    es = capsule.get("epistemic_state", {})

    def _extract_value(field: str) -> float:
        if not isinstance(es, dict):
            return 0.0
        val = es.get(field)
        if isinstance(val, dict):
            return float(val.get("value", 0.0) or 0.0)
        if isinstance(val, (int, float)):
            return float(val)
        return 0.0

    features: list[float] = []
    features.append(
        _extract_value("ontic_compatibility") or _extract_value("ontology_compatibility")
    )
    features.append(_extract_value("evidential_anchoring"))
    features.append(_extract_value("transformation_pressure"))
    features.append(
        _extract_value("redundancy_effective_n") or _extract_value("independent_redundancy")
    )
    features.append(_extract_value("uncertainty"))

    rels = capsule.get("relations", [])
    features.append(float(len(rels) if isinstance(rels, list) else 0))

    provenance = capsule.get("provenance", [])
    if isinstance(provenance, list):
        provenance_count = len(provenance)
    elif isinstance(provenance, dict):
        sources = provenance.get("sources", [])
        provenance_count = len(sources) if isinstance(sources, list) else 0
    else:
        provenance_count = 0
    features.append(float(provenance_count))

    context = capsule.get("context", {})
    geom = capsule.get("geometry", {})
    has_context_geometry = isinstance(context, dict) and bool(context.get("experimental_geometry"))
    has_legacy_geometry = isinstance(geom, dict) and bool(geom.get("enabled"))
    features.append(1.0 if has_context_geometry or has_legacy_geometry else 0.0)
    return features
