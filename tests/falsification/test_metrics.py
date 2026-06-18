from __future__ import annotations

from pathlib import Path

from hcaps.falsification.metrics import compute_dataset_metrics
from hcaps.falsification.runner import load_capsules
from hcaps.training_bridge.examples import render_flat_text

EXAMPLES = Path("examples/axf/v0_1")


def test_metrics_compute_dataset_counts() -> None:
    capsules, _, _ = load_capsules(EXAMPLES / "full_capsules.axc")
    rendered = [render_flat_text(capsule) for capsule in capsules]

    metrics = compute_dataset_metrics(capsules, rendered)

    assert metrics["record_count"] == 2
    assert metrics["relation_count"] == 2
    assert metrics["provenance_source_count"] == 2
    assert metrics["unique_claim_family_count"] == 2


def test_metrics_detect_empty_rendered_text() -> None:
    metrics = compute_dataset_metrics([{"capsule_id": "cap_empty"}], [""])

    assert metrics["record_count"] == 1
    assert metrics["empty_text_count"] == 1
