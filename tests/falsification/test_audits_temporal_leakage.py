from __future__ import annotations

from pathlib import Path

from hcaps.falsification.audits import audit_temporal_leakage
from hcaps.falsification.runner import load_capsules

EXAMPLES = Path("examples/axf/v0_1")


def test_valid_example_has_no_temporal_leakage() -> None:
    capsules, _, _ = load_capsules(EXAMPLES / "minimal_dataset.axc")

    assert audit_temporal_leakage(capsules) == []


def test_future_leakage_fixture_produces_finding() -> None:
    capsules, _, _ = load_capsules(EXAMPLES / "invalid" / "future_leakage_capsule.axc")

    findings = audit_temporal_leakage(capsules)

    assert len(findings) == 1
    assert findings[0].audit == "temporal_leakage"
    assert findings[0].capsule_id == "cap_invalid_future_leakage_001"
