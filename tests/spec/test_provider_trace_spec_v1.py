from __future__ import annotations

from tests.spec._helpers import assert_contains_all, read_spec


def test_provider_trace_fields_are_present_and_secret_safe() -> None:
    text = read_spec("PROVIDER_TRACE_V1.md")

    assert_contains_all(
        text,
        [
            "provider_trace_id",
            "provider_id",
            "provider_family",
            "provider_mode",
            "provider_config_hash",
            "prompt_template_id",
            "prompt_template_version",
            "request_hash",
            "response_hash",
            "cache_key",
            "replay_key",
            "cascade_stage",
            "escalation_reason",
            "merge_strategy",
            "merge_confidence",
            "disagreement_score",
            "disagreement_set_ref",
            "API key values",
            "bearer tokens",
            "must not be recorded",
        ],
    )


def test_provider_trace_maps_to_axt_provider_context() -> None:
    text = read_spec("PROVIDER_TRACE_V1.md")

    assert_contains_all(
        text,
        [
            "provider_id_idx",
            "provider_family_idx",
            "provider_mode_idx",
            "cascade_stage_idx",
            "escalation_reason_idx",
            "merge_confidence",
            "disagreement_score",
        ],
    )
