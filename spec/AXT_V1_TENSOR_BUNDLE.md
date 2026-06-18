# AXT v1 Tensor Bundle Contract

AXT v1 is the compiled tensor interface consumed by Axiom's structured-native
model. P2 specifies the contract; P3 implements the compiler.

AXT is not a prompt format and not a token-only dataset. It contains structured
input tensors, structured targets, availability masks, loss masks, split masks,
and metadata.

## Tensor Groups

Required groups:

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

Each group must declare:

- tensor name;
- dtype;
- shape convention;
- ragged representation strategy;
- padding strategy;
- mask strategy;
- source AXC fields;
- target AXC-out fields where applicable;
- predictor-visible or target-only status.

## Ragged Data Strategy

Variable-length structures use:

```text
values tensor + offsets tensor + mask tensor
```

This applies to:

- source spans;
- provenance lists;
- relation lists;
- relation neighborhoods;
- negative pools;
- synthetic views;
- text tokens.

Offsets are int64, masks are boolean, and padded values must be ignored wherever
the corresponding mask is false.

## Text Projection Tensors

Text projection is mandatory but secondary.

Required tensors:

- `text_input_ids`
- `text_attention_mask`
- `text_labels`
- `text_loss_mask`
- `text_render_mode`
- `special_token_registry`

Text render modes:

- `flat_text`
- `structured_text`
- `capsule_text`
- `text_projection_from_structured_state`

Special tokens are projection/baseline tools only. They are not the native AXF
substrate.

## Provider Context Tensors

P1 provider traces map into:

- `provider_id_idx`
- `provider_family_idx`
- `provider_mode_idx`
- `cascade_stage_idx`
- `escalation_reason_idx`
- `merge_confidence`
- `disagreement_score`
- optional `provider_vote_distribution`

Provider identity/slant is lateral context and must be ablatable.

## Target Tensors

The `targets` group reserves:

- `target_claim_state`
- `target_relation_type`
- `target_provenance_source`
- `target_evidence_span`
- `target_stability_status`
- `target_uncertainty`
- `target_future_summary`
- `target_temporal_state`
- `target_geometry_observables`
- `target_axc_out_field`

Missing targets are masked, not converted to negative labels.

## Masks

AXT v1 must include the mask concepts defined in
[LOSS_AND_TARGET_MASKS_V1.md](LOSS_AND_TARGET_MASKS_V1.md).

Temporal masks must prevent future-facing fields from entering predictor-side
tensors.

## Geometry Slots

Reserved geometry tensors:

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

Raw connection matrices are not semantic tensors.
