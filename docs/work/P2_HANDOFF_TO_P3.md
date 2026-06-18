# P2 Handoff To P3

Status: Consumed by P3; retained as the P2-to-P3 contract record.

P3 implemented the compiler from AXC/AXP inputs into AXT tensor bundles using
the v1 contracts in `spec/`. Future agents should use
[P3_AXT_COMPILER_HANDOFF_TO_P4.md](P3_AXT_COMPILER_HANDOFF_TO_P4.md) for the
current handoff.

## Input Locations

Tiny fixture corpus:

```text
examples/corpus/ml_software_benchmarks/
```

Generated local artifacts:

```text
artifacts/examples/ml_software_benchmarks/
  capsules.axc
  source_registry.jsonl
  claim_families.jsonl
  relation_candidates.jsonl
  negative_pools.jsonl
  extraction_trace.jsonl
  provider_disagreements.jsonl
  gold_candidates.jsonl
  dataset.axp/
```

AXP package inputs:

```text
dataset.axp/data/capsules.axc
dataset.axp/data/sources.axsrc
dataset.axp/data/relations.axr
dataset.axp/data/contexts.axctx
dataset.axp/manifests/provider_report.json
dataset.axp/manifests/license_report.json
dataset.axp/manifests/quality_report.json
```

P3 should also support the v1 preferred layout in
`spec/AXP_V1_PACKAGE_LAYOUT.md`.

## Specs To Load

- `spec/AXF_V1.md`
- `spec/AXC_V1_CAPSULE_SCHEMA.md`
- `spec/AXP_V1_PACKAGE_LAYOUT.md`
- `spec/AXT_V1_TENSOR_BUNDLE.md`
- `spec/AXC_OUT_V1_SCHEMA.md`
- `spec/FIELD_REGISTRY_V1.md`
- `spec/VOCABULARY_REGISTRY_V1.md`
- `spec/LOSS_AND_TARGET_MASKS_V1.md`
- `spec/NEGATIVE_SAMPLING_V1.md`
- `spec/PROVIDER_TRACE_V1.md`
- `spec/INTERPRETER_BOUNDARIES_V1.md`

## AXC To AXT Mapping

Use `spec/FIELD_REGISTRY_V1.md` as the source of truth. Required AXT tensor
groups:

- `ids`
- `claim`
- `temporal`
- `lateral_context`
- `provider_context`
- `provenance`
- `relations`
- `relation_neighborhoods`
- `epistemic_state`
- `negative_samples`
- `geometry_observables`
- `text_projection`
- `targets`
- `availability_masks`
- `loss_masks`
- `split_masks`
- `metadata`

Ragged values use `values + offsets + mask`.

## Targets And Masks

Compile availability and loss masks from
`spec/LOSS_AND_TARGET_MASKS_V1.md`.

Required loss masks:

- `loss_mask_text_projection`
- `loss_mask_relation_prediction`
- `loss_mask_provenance_recovery`
- `loss_mask_stability_prediction`
- `loss_mask_uncertainty_calibration`
- `loss_mask_future_summary`
- `loss_mask_temporal_prediction`
- `loss_mask_geometry_observables`
- `loss_mask_provider_context`
- `loss_mask_negative_sampling`

Required availability masks:

- `available_relation_targets`
- `available_provenance_targets`
- `available_epistemic_targets`
- `available_future_targets`
- `available_geometry_targets`
- `available_text_targets`
- `available_evaluation_references`

Missing target state is not a negative target.

## Negative Pools

Compile P1 negative pools using `spec/NEGATIVE_SAMPLING_V1.md`.

Required negative types:

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

## Provider Traces

Provider traces compile into provider/lateral-context tensors:

- `provider_id_idx`
- `provider_family_idx`
- `provider_mode_idx`
- `cascade_stage_idx`
- `escalation_reason_idx`
- `merge_confidence`
- `disagreement_score`

No API keys or secret values may enter AXT.

## Text Projection Tensors

Create:

- `text_input_ids`
- `text_attention_mask`
- `text_labels`
- `text_loss_mask`
- `text_render_mode`
- `special_token_registry`

Projection modes:

- `flat_text`
- `structured_text`
- `capsule_text`
- `text_projection_from_structured_state`

Text projection is secondary and must exclude future/evaluation-only fields.

## Geometry Slots

Reserve:

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

Do not compile raw connection matrices as semantic tensors.

## AXC-out Targets For Later Plans

P3 target tensors must align with:

- `raw_emission`
- `validated_axc_out`
- `interpreted_projection`
- `text_projection`

The compiler should preserve target paths for relation, provenance, epistemic,
temporal, geometry, and text projection outputs.

## Metadata-Only Fields

Provider raw responses, cache debug payloads, construction reports, license
reports, quality reports, and evaluation references are metadata unless the
field registry marks a derived field as predictor-visible.

## Must Never Enter Predictor Text

```text
future_summary
target_timestamp
target_only
gold_reference
evaluation_reference
answer_key
ground_truth
```

## Checks Run In P2

Record the actual P2 quality-gate results in the commit/PR final message.
