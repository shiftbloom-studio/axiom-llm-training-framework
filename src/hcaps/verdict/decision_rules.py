"""Configurable P6 decision rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

from hcaps.verdict.claims_ladder import claims_ladder_payload

VerdictOutcome = Literal[
    "proceed", "proceed_with_caution", "redesign", "kill_branch", "inconclusive"
]


class VerdictThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_relative_improvement: float = 0.05
    minimum_absolute_improvement: float = 0.01
    max_fairness_parameter_delta: float = 0.05
    max_compute_delta: float = 0.10
    kill_absolute_regression: float = -0.20
    required_controls: list[str] = Field(
        default_factory=lambda: [
            "geometry_off",
            "context_shuffle",
            "popularity_frequency_control",
            "temporal_leakage_audit",
        ]
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"verdict config must be a YAML mapping: {path}")
        return cls.model_validate(payload)


def generate_verdict(
    scores: dict[str, Any],
    *,
    fairness_report: dict[str, Any] | None = None,
    thresholds: VerdictThresholds | None = None,
) -> dict[str, Any]:
    """Generate branch-level P6 verdicts from scores and fairness controls."""

    threshold = thresholds or VerdictThresholds()
    arms = scores.get("arms", {})
    arms = arms if isinstance(arms, dict) else {}
    fairness_ok = bool(
        fairness_report.get("headline_comparison_allowed", True) if fairness_report else True
    )
    branch_verdicts = {
        "structured_native_io": _structured_native_verdict(arms, threshold, fairness_ok),
        "relation_conditioning": _ablation_verdict(
            arms,
            positive="D_structured_native_no_geometry",
            control="G_structured_native_no_relations",
            metric="relation_macro_f1",
            thresholds=threshold,
            fairness_ok=fairness_ok,
        ),
        "provenance_conditioning": _ablation_verdict(
            arms,
            positive="D_structured_native_no_geometry",
            control="F_structured_native_no_provenance",
            metric="source_recall_at_1",
            thresholds=threshold,
            fairness_ok=fairness_ok,
        ),
        "geometry_branch": _geometry_verdict(arms, threshold, fairness_ok),
        "epistemic_router": _presence_verdict(arms, "structured_epistemic_score", fairness_ok),
        "text_projection": _text_projection_verdict(arms, threshold),
        "provider_context_features": _ablation_verdict(
            arms,
            positive="E_structured_native_geometry",
            control="I_structured_native_provider_shuffle",
            metric="structured_epistemic_score",
            thresholds=threshold,
            fairness_ok=fairness_ok,
        ),
    }
    reached_level = _claims_level(arms, branch_verdicts, fairness_ok)
    return {
        "schema_version": "p6-verdict-0.1.0",
        "branch_verdicts": branch_verdicts,
        "claims_ladder": claims_ladder_payload(reached_level),
        "fairness": fairness_report or {},
        "thresholds": threshold.model_dump(mode="json"),
        "no_hkr_proof_claim": True,
        "external_llm_judging_used": False,
        "overall_verdict": _overall(branch_verdicts),
    }


def _structured_native_verdict(
    arms: dict[str, Any],
    thresholds: VerdictThresholds,
    fairness_ok: bool,
) -> dict[str, Any]:
    native = _metric(arms, "D_structured_native_no_geometry", "structured_epistemic_score")
    structured_text = _metric(arms, "B_structured_text", "structured_epistemic_score")
    return _delta_verdict(
        native - structured_text,
        thresholds=thresholds,
        fairness_ok=fairness_ok,
        evidence="structured_native_no_geometry minus structured_text",
    )


def _geometry_verdict(
    arms: dict[str, Any],
    thresholds: VerdictThresholds,
    fairness_ok: bool,
) -> dict[str, Any]:
    geometry = _metric(arms, "E_structured_native_geometry", "structured_epistemic_score")
    off = _metric(arms, "D_structured_native_no_geometry", "structured_epistemic_score")
    shuffle = _metric(arms, "H_structured_native_context_shuffle", "structured_epistemic_score")
    popularity = _metric(arms, "J_popularity_frequency_control", "structured_epistemic_score")
    delta = geometry - off
    degradation = geometry - shuffle
    popularity_delta = geometry - popularity
    verdict = _delta_verdict(
        delta,
        thresholds=thresholds,
        fairness_ok=fairness_ok,
        evidence="geometry_on minus geometry_off",
    )
    verdict["control_effects"] = {
        "context_shuffle_degradation": degradation,
        "geometry_signal_vs_popularity_control": popularity_delta,
    }
    if verdict["outcome"] == "proceed" and degradation < thresholds.minimum_absolute_improvement:
        verdict["outcome"] = "proceed_with_caution"
        verdict["reason"] = "geometry improved but context-shuffle degradation is weak"
    return verdict


def _ablation_verdict(
    arms: dict[str, Any],
    *,
    positive: str,
    control: str,
    metric: str,
    thresholds: VerdictThresholds,
    fairness_ok: bool,
) -> dict[str, Any]:
    if positive not in arms or control not in arms:
        return {"outcome": "inconclusive", "reason": "required ablation arm was not run"}
    delta = _metric(arms, positive, metric) - _metric(arms, control, metric)
    return _delta_verdict(
        delta,
        thresholds=thresholds,
        fairness_ok=fairness_ok,
        evidence=f"{positive} minus {control} on {metric}",
    )


def _presence_verdict(arms: dict[str, Any], metric: str, fairness_ok: bool) -> dict[str, Any]:
    if not fairness_ok:
        return {"outcome": "inconclusive", "reason": "fairness report blocks headline claim"}
    if not arms:
        return {"outcome": "inconclusive", "reason": "no scored arms"}
    best = max(_metric(payload, metric) for payload in arms.values() if isinstance(payload, dict))
    if best > 0.0:
        return {"outcome": "proceed_with_caution", "reason": "runtime metric exists", "best": best}
    return {"outcome": "redesign", "reason": "runtime metric is absent or zero", "best": best}


def _text_projection_verdict(
    arms: dict[str, Any],
    thresholds: VerdictThresholds,
) -> dict[str, Any]:
    best = max(
        (
            _metric(payload, "token_accuracy")
            for payload in arms.values()
            if isinstance(payload, dict)
        ),
        default=0.0,
    )
    if best >= thresholds.minimum_absolute_improvement:
        return {
            "outcome": "proceed_with_caution",
            "reason": "secondary text projection is viable",
            "best": best,
        }
    return {
        "outcome": "redesign",
        "reason": "text projection is too weak in this run",
        "best": best,
    }


def _delta_verdict(
    delta: float,
    *,
    thresholds: VerdictThresholds,
    fairness_ok: bool,
    evidence: str,
) -> dict[str, Any]:
    if not fairness_ok:
        return {
            "outcome": "inconclusive",
            "reason": "fairness report blocks headline claim",
            "delta": delta,
        }
    if delta <= thresholds.kill_absolute_regression:
        return {
            "outcome": "kill_branch",
            "reason": "large controlled regression",
            "delta": delta,
            "evidence": evidence,
        }
    if delta >= thresholds.minimum_absolute_improvement:
        outcome: VerdictOutcome = "proceed"
    elif delta > 0.0:
        outcome = "proceed_with_caution"
    elif delta > -thresholds.minimum_absolute_improvement:
        outcome = "inconclusive"
    else:
        outcome = "redesign"
    return {"outcome": outcome, "reason": evidence, "delta": delta}


def _claims_level(
    arms: dict[str, Any],
    branch_verdicts: dict[str, dict[str, Any]],
    fairness_ok: bool,
) -> int:
    if not arms:
        return 0
    level = 1
    if _metric(arms, "B_structured_text", "structured_epistemic_score") > _metric(
        arms,
        "A_flat_text",
        "structured_epistemic_score",
    ):
        level = 2
    if fairness_ok and branch_verdicts["structured_native_io"]["outcome"] in {
        "proceed",
        "proceed_with_caution",
    }:
        level = max(level, 3)
    if branch_verdicts["geometry_branch"]["outcome"] in {"proceed", "proceed_with_caution"}:
        level = max(level, 4)
    return level


def _overall(branch_verdicts: dict[str, dict[str, Any]]) -> VerdictOutcome:
    outcomes = [str(value.get("outcome", "inconclusive")) for value in branch_verdicts.values()]
    if any(outcome == "kill_branch" for outcome in outcomes):
        return "redesign"
    if any(outcome == "redesign" for outcome in outcomes):
        return "redesign"
    if any(outcome == "proceed" for outcome in outcomes):
        return "proceed_with_caution"
    return "inconclusive"


def _metric(arms_or_payload: dict[str, Any], arm_id_or_key: str, key: str | None = None) -> float:
    if key is None:
        value = arms_or_payload.get(arm_id_or_key, 0.0)
        return float(value) if isinstance(value, int | float) else 0.0
    payload = arms_or_payload.get(arm_id_or_key, {})
    if not isinstance(payload, dict):
        return 0.0
    value = payload.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0
