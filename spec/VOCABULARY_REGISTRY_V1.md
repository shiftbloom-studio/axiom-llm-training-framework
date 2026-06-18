# AXF v1 Vocabulary Registry

P3 must snapshot and hash the vocabularies used for tensor compilation. New
values may be appended in later versions, but existing ids must remain stable
within a compiled AXT bundle.

## Claim Types

- `scientific_claim`
- `causal_claim`
- `measurement_claim`
- `method_claim`
- `definitional_claim`
- `historical_claim`
- `other`

## Relation Types

- `supports`
- `contradicts`
- `extends`
- `refines`
- `supersedes`
- `is_replicated_by`
- `uses_method_from`
- `shares_evidence_with`
- `same_claim_family_as`
- `near_but_distinct_from`
- `mentions`
- `related`

## Status Labels

- `unassessed`
- `emerging`
- `contested`
- `supported`
- `established`
- `superseded`
- `retracted`
- `fragmented`
- `field_dependent`
- `historically_plausible`
- `unknown`

## Provider Families And Modes

Provider families:

- `deterministic`
- `openai_compatible`
- `python_callable`
- `human`
- `custom`

Provider modes:

- `deterministic`
- `local`
- `remote`
- `human`
- `custom`

## Cascade Stages And Escalation Reasons

Cascade stages:

- `primary`
- `gate`
- `escalation`
- `merge`
- `fallback`
- `human_review`

Escalation reasons:

- `low_confidence`
- `high_impact_claim_type`
- `provider_warning`
- `relation_ambiguity`
- `low_source_grounding`
- `provider_disagreement`
- `manual_review`

## Context Fields

- `field`
- `subfield`
- `community`
- `method_context`
- `venue_context`
- `language_register`
- `source_context`
- `extractor_context`
- `provider_context`

## Negative Sample Types

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

## Geometry Observable Names

- `geometry_enabled`
- `context_node_ids`
- `context_edge_ids`
- `context_loop_ids`
- `transport_path_ids`
- `curvature_score`
- `holonomy_norm`
- `trace_summary`
- `spectrum_summary`
- `context_lability`
- `geometry_loss_mask`
- `geometry_ablation_mask`

## Text Projection Modes

- `flat_text`
- `structured_text`
- `capsule_text`
- `text_projection_from_structured_state`

## Special Tokens For Text Projection

Special tokens are permitted only for text projection and text-rendered
baselines. They are not the native Axiom substrate.

- `<AX_CLAIM>`
- `<AX_PROVENANCE>`
- `<AX_RELATION>`
- `<AX_TEMPORAL>`
- `<AX_CONTEXT>`
- `<AX_EPISTEMIC>`
- `<AX_GEOMETRY>`
- `<AX_VIEW>`
- `<AX_END>`
