# AXC v1 Capsule Schema

AXC v1 is the canonical claim-state capsule stream for Axiom. It is structured
claim-field data, not decorated prompt text.

Each record must include:

```text
format = "AXC"
format_version = "1.0.0"
```

## Required Sections

- `ids`
- `claim`
- `surface_forms`
- `temporal`
- `lateral_context`
- `provider_context`
- `epistemic_state`
- `relations`
- `provenance`
- `negative_pools`
- `evaluation_references`
- `synthetic_views`
- `geometry`
- `training`
- `quality`
- `lineage`

## Identity Fields

`ids` must distinguish:

- `capsule_id`
- `claim_family_id`
- `claim_state_id`
- `context_id`
- `source_ids`
- `provider_trace_ids`
- `relation_ids`
- `negative_pool_ids`
- `evaluation_reference_ids`

## Temporal Section

Time is the claim-state drift axis. It must not be flattened into lateral
context.

Required temporal fields:

- `valid_as_of`
- `observed_at`
- `constructed_at`
- `source_publication_date`
- `retrieved_at`
- `temporal_cutoff_policy`
- `target_only`
- `future_target`
- `temporal_split_eligibility`

Future-facing values must not enter predictor text or predictor-side AXT tensor
groups.

## Lateral Context Section

`lateral_context` records context around the claim core:

- `field`
- `subfield`
- `community`
- `method_context`
- `venue_context`
- `language_register`
- `source_context`
- `extractor_context`
- `provider_context_ref`

Provider identity/slant is lateral context. Time is not.

## Provider Context References

`provider_context` references P1 construction traces. It may include:

- `provider_id`
- `provider_family`
- `provider_mode`
- `provider_config_hash`
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

Secrets are forbidden.

## Epistemic State

Each epistemic proxy must contain:

- `value`
- `method`
- `confidence`
- `basis`
- `source_fields_used`
- optional `provider_trace_ref`

Required slots:

- `ontology_compatibility`
- `ontic_compatibility`
- `evidential_anchoring`
- `transformation_pressure`
- `independent_redundancy`
- `uncertainty`
- `stability_status`

Independent redundancy is not popularity or raw frequency.

## Negative Pools

AXC v1 may reference typed negative pools. Supported types:

- `near_topic_negative`
- `near_claim_negative`
- `same_context_unrelated`
- `same_source_unrelated`
- `contradiction_candidate`
- `supersession_candidate`
- `temporal_negative`
- `provider_disagreement_negative`
- `hard_relation_negative`
- `provenance_negative`

Each item must define:

- `negative_id`
- `type`
- `target_ref` or `source_ref`
- `sampling_method`
- `sampling_basis`
- `hardness_score`
- optional `confidence`
- optional `provider_trace_ref`

## Evaluation References

Evaluation hooks must remain evaluation-only:

- `evaluation_reference_id`
- `reference_type`
- `reference_span_ref`
- optional `reference_relation_type`
- optional `reference_status`
- optional `human_verified`
- optional `verification_method`
- `allowed_split`
- `not_predictor_visible`

They must not be rendered into predictor text.

## Geometry

The `geometry` section reserves gauge-invariant observable slots only. Raw
connection matrices and basis-dependent parameters are not valid AXC fields.

## Training And Mask Hints

`training` may declare target availability, split eligibility, and mask hints.
Missing targets are masked and are not negative examples.
