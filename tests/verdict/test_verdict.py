from __future__ import annotations

from hcaps.verdict import VerdictThresholds, generate_verdict
from hcaps.verdict.claims_ladder import claims_ladder_payload


def test_verdict_rules_emit_branch_outcomes() -> None:
    thresholds = VerdictThresholds(minimum_absolute_improvement=0.01)
    scores = {
        "arms": {
            "A_flat_text": {"structured_epistemic_score": 0.1, "token_accuracy": 0.1},
            "B_structured_text": {"structured_epistemic_score": 0.2, "token_accuracy": 0.1},
            "D_structured_native_no_geometry": {
                "structured_epistemic_score": 0.4,
                "relation_macro_f1": 0.5,
                "source_recall_at_1": 0.5,
                "token_accuracy": 0.1,
            },
            "E_structured_native_geometry": {"structured_epistemic_score": 0.5},
            "H_structured_native_context_shuffle": {"structured_epistemic_score": 0.3},
            "J_popularity_frequency_control": {"structured_epistemic_score": 0.2},
        }
    }
    verdict = generate_verdict(
        scores,
        fairness_report={"headline_comparison_allowed": True},
        thresholds=thresholds,
    )

    assert verdict["branch_verdicts"]["structured_native_io"]["outcome"] == "proceed"
    assert verdict["branch_verdicts"]["geometry_branch"]["outcome"] == "proceed"


def test_verdict_rules_can_redesign_kill_and_inconclusive() -> None:
    thresholds = VerdictThresholds(minimum_absolute_improvement=0.1, kill_absolute_regression=-0.2)
    blocked = generate_verdict(
        {"arms": {}},
        fairness_report={"headline_comparison_allowed": False},
        thresholds=thresholds,
    )
    assert blocked["branch_verdicts"]["structured_native_io"]["outcome"] == "inconclusive"

    killed = generate_verdict(
        {
            "arms": {
                "B_structured_text": {"structured_epistemic_score": 0.8},
                "D_structured_native_no_geometry": {"structured_epistemic_score": 0.1},
            }
        },
        fairness_report={"headline_comparison_allowed": True},
        thresholds=thresholds,
    )
    assert killed["branch_verdicts"]["structured_native_io"]["outcome"] == "kill_branch"


def test_claims_ladder_never_emits_hkr_proof_language() -> None:
    payload = claims_ladder_payload(6)

    assert payload["proof_language_emitted"] is False
    assert payload["forbidden_level_7"] == "HKR is proven."
