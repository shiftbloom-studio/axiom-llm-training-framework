from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest

from hcaps.format.identifiers import (
    axc_id,
    claim_family_id,
    claim_state_id,
    context_id,
    relation_id,
    source_id,
    span_id,
)


@pytest.fixture(scope="session")
def axf_examples_dir() -> Path:
    return Path(__file__).parents[2] / "examples" / "axf" / "v0_1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


@pytest.fixture
def minimal_axc_record() -> dict[str, Any]:
    src_id = source_id("paper", "fixture-source")
    spn_id = span_id("fixture-span")
    family_id = claim_family_id("aspirin-cox", "aspirin inhibits cox")
    state_id = claim_state_id("2024-01-01", "aspirin inhibits cox", "2024-01-01")
    return {
        "format": "AXC",
        "format_version": "0.1.0",
        "ids": {
            "capsule_id": axc_id(family_id, state_id),
            "claim_family_id": family_id,
            "claim_state_id": state_id,
            "context_id": context_id("medicine"),
        },
        "claim": {
            "canonical_text": "Aspirin inhibits cyclooxygenase enzymes.",
            "claim_type": "method_claim",
            "language": "en",
            "family_label": "biochemistry",
        },
        "surface_forms": {
            "canonical": "Aspirin inhibits cyclooxygenase enzymes.",
            "source_spans": [
                {
                    "span_id": spn_id,
                    "source_id": src_id,
                    "text": "Aspirin inhibits cyclooxygenase enzymes.",
                    "start_char": 0,
                    "end_char": 42,
                    "source_date": "2023-01-01T00:00:00Z",
                }
            ],
            "normalized_views": {"primary": "Aspirin inhibits cyclooxygenase enzymes."},
            "generated_views": {},
        },
        "temporal": {
            "valid_as_of": "2024-01-01T00:00:00Z",
            "observed_at": "2024-01-01T00:00:00Z",
            "constructed_at": "2024-02-01T00:00:00Z",
            "source_publication_date": "2023-01-01T00:00:00Z",
            "cutoff_policy": "predictor_side_sources_must_not_exceed_valid_as_of",
            "leakage_validation_status": "passed",
        },
        "context": {"domains": ["medicine"], "communities": ["clinical"]},
        "epistemic_state": {
            "status": "unassessed",
            "ontology_compatibility": {"value": 0.5, "method": "fixture"},
            "evidential_anchoring": {"value": 0.6, "method": "fixture"},
            "transformation_pressure": {"value": 0.1, "method": "fixture"},
            "uncertainty": {"value": 0.4, "method": "fixture"},
            "independent_redundancy": {
                "effective_count": 2.5,
                "method": "independent_source_proxy",
                "confidence": 0.5,
            },
            "status_trajectory": ["emerging"],
        },
        "relations": [
            {
                "relation_id": relation_id("fixture-relation"),
                "target_claim_family_id": claim_family_id("cox", "cox enzymes"),
                "relation_type": "related",
                "confidence": 0.4,
                "evidence_span_ids": [spn_id],
                "extraction_method": "fixture",
            }
        ],
        "geometry": {
            "enabled": False,
            "gauge_policy": "gauge_invariant_observables_only",
            "curvature_score": None,
            "transport_observables": {},
            "loop_identifiers": [],
            "context_sequence": [],
        },
        "provenance": {
            "sources": [
                {
                    "source_id": src_id,
                    "title": "Fixture source",
                    "path": "fixture.txt",
                    "source_date": "2023-01-01T00:00:00Z",
                    "license": "fixture",
                }
            ],
            "construction_method": "fixture",
            "extractor": "fixture",
        },
        "training": {
            "eligible": True,
            "allowed_splits": ["train"],
            "target_fields": ["surface_forms.canonical"],
            "future_label_fields": [],
        },
        "quality": {"extraction_confidence": 0.8, "validation_notes": [], "warnings": []},
    }
