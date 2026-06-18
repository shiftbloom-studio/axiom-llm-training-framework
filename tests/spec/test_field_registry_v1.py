from __future__ import annotations

from tests.spec._helpers import ROOT, assert_contains_all, read_spec


def test_field_registry_maps_required_p1_outputs_to_axt_and_axc_out() -> None:
    text = read_spec("FIELD_REGISTRY_V1.md")

    assert_contains_all(
        text,
        [
            "claim_families",
            "provider_context",
            "negative_pools",
            "evaluation_references",
            "AXT tensor group",
            "AXC-out target path",
            "visibility",
            "mask behavior",
            "predictor",
            "target_only",
            "metadata",
            "evaluation_only",
        ],
    )


def test_p3_handoff_exists_and_points_to_field_registry() -> None:
    handoff = (ROOT / "docs" / "work" / "P2_HANDOFF_TO_P3.md").read_text(encoding="utf-8")

    assert "spec/FIELD_REGISTRY_V1.md" in handoff
    assert "ids" in handoff
    assert "provider_context" in handoff
    assert "negative_samples" in handoff
