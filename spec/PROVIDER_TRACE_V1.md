# AXF v1 Provider Trace Contract

Provider traces record data-construction provenance. They are never training,
evaluation, scoring, or benchmark-judging calls.

## Required Fields

- `provider_trace_id`
- `provider_id`
- `provider_family`
- `provider_mode`
- `provider_config_hash`
- `model_name`
- `prompt_template_id`
- `prompt_template_version`
- `request_hash`
- `response_hash`
- `cache_key`
- `replay_key`
- `cascade_stage`
- `escalation_reason`
- `merge_strategy`
- `merge_confidence`
- `disagreement_score`
- `disagreement_set_ref`
- `created_at`
- `input_hash`
- `output_hash`
- `warnings`

## Secret Safety

Forbidden in provider traces:

- API key values;
- bearer tokens;
- passwords;
- raw credential headers;
- provider account secrets.

Environment variable names such as `AXIOM_LOCAL_PROVIDER_KEY` may be recorded as
configuration provenance. Their values must not be recorded.

## AXT Mapping

Provider trace metadata may compile into:

- `provider_id_idx`
- `provider_family_idx`
- `provider_mode_idx`
- `cascade_stage_idx`
- `escalation_reason_idx`
- `merge_confidence`
- `disagreement_score`
- optional `provider_vote_distribution`

Provider identity/slant is lateral context and must be ablatable.
