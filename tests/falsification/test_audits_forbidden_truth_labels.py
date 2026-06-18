from __future__ import annotations

from pathlib import Path

from hcaps.falsification.audits import audit_forbidden_truth_labels
from hcaps.falsification.runner import load_capsules

EXAMPLES = Path("examples/axf/v0_1")


def test_forbidden_truth_label_fixture_produces_finding() -> None:
    capsules, _, _ = load_capsules(EXAMPLES / "invalid" / "forbidden_truth_label_capsule.axc")

    findings = audit_forbidden_truth_labels(capsules)

    assert len(findings) == 1
    assert findings[0].value == "ground_truth"


def test_recursive_nested_truth_labels_are_found() -> None:
    capsules = [
        {
            "capsule_id": "cap_nested_truth",
            "claim": {"claim_id": "claim_nested", "canonical_text": "Nested labels are forbidden."},
            "surface_forms": {"primary_text": "Nested labels are forbidden."},
            "epistemic_state": {},
            "provenance": [],
            "context": {},
            "metadata": {"deep": {"is_correct": False}},
        }
    ]

    findings = audit_forbidden_truth_labels(capsules)

    assert len(findings) == 1
    assert findings[0].path == "metadata.deep.is_correct"
