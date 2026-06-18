from __future__ import annotations

from tests.spec._helpers import assert_contains_all, read_spec


def test_axc_out_schema_preserves_required_layers() -> None:
    text = read_spec("AXC_OUT_V1_SCHEMA.md")

    assert_contains_all(
        text,
        [
            "raw_emission",
            "validated_axc_out",
            "interpreted_projection",
            "text_projection",
            "claim_state_prediction",
            "relation_predictions",
            "provenance_predictions",
            "epistemic_predictions",
            "temporal_predictions",
            "geometry_observables",
        ],
    )


def test_interpreter_boundaries_prevent_hidden_oracle_repairs() -> None:
    text = read_spec("INTERPRETER_BOUNDARIES_V1.md")

    assert_contains_all(
        text,
        [
            "add evidence not emitted by the model",
            "hidden graph oracle",
            "inject future information",
            "hide raw invalid emissions",
            "score interpreter repairs as model competence",
        ],
    )
