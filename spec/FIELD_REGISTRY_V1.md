# AXF v1 Field Registry

This registry tells P3 how to compile P1/P2 fields without guessing semantics.

Columns:

```text
field_name | semantic_owner | source_format | AXC path | AXT tensor group |
AXC-out target path | visibility | requiredness | mask behavior | dtype |
allowed values / vocabulary | hash behavior
```

## Required Field Mappings

| field_name | semantic_owner | source_format | AXC path | AXT tensor group | AXC-out target path | visibility | requiredness | mask behavior | dtype | vocabulary | hash behavior |
|---|---|---|---|---|---|---|---|---|---|---|---|
| capsule_id | identity | AXC | ids.capsule_id | ids | validated_axc_out.claim_state_prediction.capsule_id | predictor | required | none | string_hash | id | included |
| claim_family_id | identity | AXC/P1 claim_families | ids.claim_family_id | ids | validated_axc_out.claim_state_prediction.claim_family_id | predictor | required | none | string_hash | id | included |
| claim_state_id | identity | AXC | ids.claim_state_id | ids | validated_axc_out.claim_state_prediction.claim_state_id | predictor | required | none | string_hash | id | included |
| context_id | identity/context | AXC/P1 contexts.axctx | ids.context_id | lateral_context | validated_axc_out.claim_state_prediction.context_id | predictor | required | none | string_hash | id | included |
| canonical_text | claim | AXC | claim.canonical_text | claim/text_projection | text_projection | predictor | required | text_loss_mask | string/token_ids | text | included |
| claim_type | claim | AXC | claim.claim_type | claim | validated_axc_out.claim_state_prediction.claim_type | predictor | required | none | int64 | claim_type | included |
| source_span | provenance | AXC | surface_forms.source_spans | provenance | validated_axc_out.provenance_predictions.spans | predictor | optional | loss_mask_evidence_span | ragged | span | included |
| synthetic_views | projection | P1 claim_families | synthetic_views | text_projection | text_projection | metadata | optional | text_loss_mask | ragged_text | text_render_mode | included |
| valid_as_of | temporal | AXC | temporal.valid_as_of | temporal | validated_axc_out.temporal_predictions.valid_as_of | predictor | required | temporal_mask | timestamp | temporal | included |
| future_target | temporal | AXC | temporal.future_target | targets | validated_axc_out.temporal_predictions.future_state | target_only | optional | loss_mask_future_summary | bool/string | target_availability | included |
| lateral_context | context | AXC/P1 source_registry | lateral_context | lateral_context | validated_axc_out.claim_state_prediction.lateral_context | predictor | required | context_mask | categorical/ragged | context_field | included |
| provider_id | provider | P1 provider traces | provider_context.provider_id | provider_context | interpreted_projection.provider_context.provider_id | metadata | optional | loss_mask_provider_context | int64 | provider_id | included |
| provider_mode | provider | P1 provider traces | provider_context.provider_mode | provider_context | interpreted_projection.provider_context.provider_mode | metadata | optional | loss_mask_provider_context | int64 | provider_mode | included |
| cascade_stage | provider | P1 cascade traces | provider_context.cascade_stage | provider_context | interpreted_projection.provider_context.cascade_stage | metadata | optional | loss_mask_provider_context | int64 | cascade_stage | included |
| merge_confidence | provider | P1 merge traces | provider_context.merge_confidence | provider_context | interpreted_projection.provider_context.merge_confidence | metadata | optional | loss_mask_provider_context | float32 | unit_float | included |
| disagreement_score | provider | P1 disagreements | provider_context.disagreement_score | provider_context | interpreted_projection.provider_context.disagreement_score | metadata | optional | loss_mask_provider_context | float32 | unit_float | included |
| relation_type | relation | AXC/P1 relations.axr | relations[].relation_type | relations | validated_axc_out.relation_predictions[].relation_type | predictor/target | optional | loss_mask_relation_prediction | int64 | relation_type | included |
| provenance_source | provenance | AXC/P1 sources.axsrc | provenance.sources | provenance | validated_axc_out.provenance_predictions.sources | predictor/target | optional | loss_mask_provenance_recovery | ragged | source_id | included |
| ontology_compatibility | epistemic | P1 proxies | epistemic_state.ontology_compatibility | epistemic_state | validated_axc_out.epistemic_predictions.ontology_compatibility | predictor/target | optional | available_epistemic_targets | float32 | proxy | included |
| evidential_anchoring | epistemic | P1 proxies | epistemic_state.evidential_anchoring | epistemic_state | validated_axc_out.epistemic_predictions.evidential_anchoring | predictor/target | optional | available_epistemic_targets | float32 | proxy | included |
| transformation_pressure | epistemic | P1 proxies | epistemic_state.transformation_pressure | epistemic_state | validated_axc_out.epistemic_predictions.transformation_pressure | predictor/target | optional | available_epistemic_targets | float32 | proxy | included |
| independent_redundancy | epistemic | P1 proxies | epistemic_state.independent_redundancy | epistemic_state | validated_axc_out.epistemic_predictions.independent_redundancy | predictor/target | optional | available_epistemic_targets | float32 | proxy | included |
| uncertainty | epistemic | P1 proxies | epistemic_state.uncertainty | epistemic_state | validated_axc_out.epistemic_predictions.uncertainty | predictor/target | optional | loss_mask_uncertainty_calibration | float32 | proxy | included |
| stability_status | epistemic | AXC/P1 proxies | epistemic_state.stability_status | targets | validated_axc_out.epistemic_predictions.stability_status | target_only | optional | loss_mask_stability_prediction | int64 | status_label | included |
| negative_pool | negatives | P1 negative_pools | negative_pools | negative_samples | raw_emission.negative_sample_scores | target_only | optional | loss_mask_negative_sampling | ragged | negative_sample_type | included |
| evaluation_reference | evaluation | P1 gold candidates | evaluation_references | metadata | interpreted_projection.evaluation_reference_ref | evaluation_only | optional | available_evaluation_references | ragged | reference_type | excluded_from_predictor_hash |
| geometry_observable | geometry | AXC/AXC-out | geometry | geometry_observables | validated_axc_out.geometry_observables | predictor/target | optional | loss_mask_geometry_observables | float32 | geometry_observable | included |
| text_projection | projection | AXC/AXC-out | surface_forms/generated_views | text_projection | text_projection | predictor/target | optional | loss_mask_text_projection | token_ids | text_render_mode | included |

## Visibility Rules

- `predictor` fields may enter predictor-side AXT tensors.
- `target_only` fields may enter target tensors and loss masks only.
- `metadata` fields may enter manifests, ablations, and context controls.
- `evaluation_only` fields must not enter predictor text or predictor tensors.

## Mask States

Allowed target state values are defined in
[LOSS_AND_TARGET_MASKS_V1.md](LOSS_AND_TARGET_MASKS_V1.md):

```text
unknown
missing
not_applicable
not_predictor_visible
target_only
masked_by_split
available
```
