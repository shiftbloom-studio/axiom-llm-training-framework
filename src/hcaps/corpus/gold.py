"""Human-review and gold-reference candidate export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from hcaps.format.package import CAPSULE_STREAM_PATH
from hcaps.format.streams import read_axc_stream
from hcaps.providers.traces import write_jsonl


def gold_candidate_records(input_path: str | Path) -> list[dict[str, Any]]:
    """Create human-review candidate records from an AXC stream or AXP package."""

    path = Path(input_path)
    stream_path = path / CAPSULE_STREAM_PATH if path.is_dir() else path
    capsules = list(read_axc_stream(stream_path))
    records: list[dict[str, Any]] = []
    for capsule in capsules:
        records.append(
            {
                "candidate_id": f"gold_{capsule.ids.claim_family_id}",
                "claim_family_id": capsule.ids.claim_family_id,
                "capsule_id": capsule.ids.capsule_id,
                "canonical_text": capsule.claim.canonical_text,
                "gold_reference": {
                    "claim_boundary_review": None,
                    "relation_review": [],
                    "provenance_span_review": [
                        span.model_dump(mode="json") for span in capsule.surface_forms.source_spans
                    ],
                    "temporal_cutoff_review": capsule.temporal.valid_as_of.isoformat(),
                    "epistemic_proxy_review": None,
                    "hard_negative_relation_examples": [],
                    "near_but_distinct_claim_examples": [],
                },
                "human_verified_reference": False,
                "evaluation_reference": False,
            }
        )
    return records


def export_gold_candidates(input_path: str | Path, output_path: str | Path) -> Path:
    """Export gold-reference candidate records as JSONL."""

    target = Path(output_path)
    write_jsonl(target, gold_candidate_records(input_path))
    return target


def read_gold_candidates(path: str | Path) -> list[dict[str, Any]]:
    """Read exported gold-reference candidates."""

    return [orjson.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]
