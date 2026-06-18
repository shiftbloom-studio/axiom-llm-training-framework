from __future__ import annotations

from tests.spec._helpers import assert_contains_all, read_spec


def test_axt_tensor_contract_includes_structured_groups_and_ragged_strategy() -> None:
    text = read_spec("AXT_V1_TENSOR_BUNDLE.md")

    assert_contains_all(
        text,
        [
            "ids",
            "claim",
            "temporal",
            "lateral_context",
            "provider_context",
            "provenance",
            "relations",
            "relation_neighborhoods",
            "epistemic_state",
            "negative_samples",
            "geometry_observables",
            "text_projection",
            "targets",
            "availability_masks",
            "loss_masks",
            "values tensor + offsets tensor + mask tensor",
        ],
    )


def test_text_projection_is_secondary_but_mandatory() -> None:
    text = read_spec("AXT_V1_TENSOR_BUNDLE.md")

    assert_contains_all(
        text,
        [
            "Text projection is mandatory but secondary",
            "text_input_ids",
            "text_attention_mask",
            "text_labels",
            "text_loss_mask",
            "text_render_mode",
            "special_token_registry",
        ],
    )
