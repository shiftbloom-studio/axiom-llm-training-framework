# P4 Handoff To P6

Status: P4 implementation-complete handoff for training and experiment runtime.

P4 implemented a forward-pass model stack and loss-ready outputs. It did not implement losses, training loops, optimizer/scheduler state, experiment orchestration, benchmark scoring, or evaluation verdicts.

## Model Output Object

`AxiomStructuredModel(batch)` returns `AxiomModelOutput`:

```text
latent_state
slot_states
raw_axc_out
text_projection_logits
auxiliary_logits
router_diagnostics
geometry_diagnostics
masks
metadata
diagnostics
```

The output is designed for P6 loss wiring. P4 does not compute final losses.

## Raw AXC-out Fields

`output.raw_axc_out.raw_emission` contains:

```text
claim_state_logits
relation_type_logits
relation_target_logits
provenance_source_logits
evidence_span_logits
epistemic_status_logits
epistemic_values
uncertainty_values
stability_logits
future_summary_latent
geometry_observable_values
masks
metadata
```

`validated_axc_out` reports shape/mask/name validation. It does not repair model emissions.

## Text Projection

`text_projection_logits` shape:

```text
[batch, text_sequence_length, text_vocab_size]
```

The head consumes P3 `text_projection.text_input_ids` when present. In `structure_only_no_text_projection` ablation mode, the field is `None`.

## Auxiliary Outputs

`auxiliary_logits` includes router head/loss-steering tensors:

```text
head_weights
relation_head_weight
provenance_head_weight
future_head_weight
geometry_head_weight
uncertainty_gate
revision_gate
```

These are routing and loss-steering signals, not truth or correctness labels.

## Expected P3 Targets

P6 should align P4 outputs with P3 `targets`:

```text
relation_target_types
relation_target_ids
provenance_target_ids
status_target_idx
uncertainty_target
redundancy_target
future_summary_target_ref
geometry_observable_targets
text_projection_targets
target_axc_out_layer_idx
```

Expected masks:

```text
loss_mask_text_projection
loss_mask_relation_prediction
loss_mask_provenance_recovery
loss_mask_stability_prediction
loss_mask_uncertainty_calibration
loss_mask_future_summary
loss_mask_temporal_prediction
loss_mask_geometry_observables
loss_mask_provider_context
loss_mask_negative_sampling
```

Missing targets must be masked, not treated as negatives.

## Ablation Modes

P4 supports:

```text
text_only
structure_only_no_text_projection
no_provenance
no_relations
no_context
no_provider_context
no_side_channels
geometry_off
router_off
relation_neighborhood_off
```

The active ablations are available at:

```text
output.diagnostics["active_ablation_modes"]
```

## Known Limitations

- P4 future summary output is a latent/logit head, not final generated text.
- P4 geometry output is shape-compatible and gauge-invariant, but P5 learned geometry is not implemented here.
- P4 serialization uses ordinary PyTorch `state_dict`; P6 may extend this for optimizer/scheduler/checkpoint runtime.
- P4 diagnostics are for development and experiment wiring, not benchmark claims.
- P4 does not call external LLMs.
