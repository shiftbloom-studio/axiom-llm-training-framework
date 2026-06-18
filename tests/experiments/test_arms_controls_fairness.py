from __future__ import annotations

from hcaps.axt import AxtBatch
from hcaps.experiments.arms import default_arm_catalog, smoke_arm_ids
from hcaps.experiments.budgets import ArmBudget
from hcaps.experiments.controls import apply_control_transform, control_manifest
from hcaps.experiments.matching import fairness_report


def test_full_and_smoke_arm_catalogs_are_expressible() -> None:
    catalog = default_arm_catalog(input_axt_path="fixture.axt", model_config="model.yaml")

    assert set(smoke_arm_ids()).issubset(catalog)
    assert "I_structured_native_provider_shuffle" in catalog
    assert catalog["E_structured_native_geometry"].geometry_mode == "learned"


def test_context_and_provider_shuffle_controls_are_represented() -> None:
    batch = AxtBatch(
        records=[
            {"lateral_context": {"x": 1}, "provider_context": {"p": 1}},
            {"lateral_context": {"x": 2}, "provider_context": {"p": 2}},
        ]
    )

    context = apply_control_transform(batch, control_transform="context_shuffle", text_mode=None)
    provider = apply_control_transform(batch, control_transform="provider_shuffle", text_mode=None)

    assert context.records[0]["lateral_context"] == {"x": 2}
    assert provider.records[0]["provider_context"] == {"p": 2}
    assert control_manifest("provider_shuffle", None)["provider_calls_allowed"] is False


def test_fairness_report_detects_parameter_mismatch() -> None:
    budgets = [
        ArmBudget("a", 100, 100, 2, 4, 16, 4, 0.0, 200.0, "cpu"),
        ArmBudget("b", 200, 200, 2, 4, 16, 4, 0.0, 400.0, "cpu"),
    ]
    report = fairness_report(
        budgets=budgets,
        source_content_id="same",
        split_id="split",
        extraction_substrate="axt",
        max_parameter_delta=0.05,
        max_compute_delta=0.05,
    )

    assert report["matched_parameter_budget"] is False
    assert report["headline_comparison_allowed"] is False
