# P3 Handoff To P4

Status: P3 implementation-complete handoff, consumed by P4.

Current next action after P4:

```text
Implement P5: Learned Geometry & Claim-Field Graph Dynamics.
```

P3 implemented the executable AXT compiler/runtime data interface. It did not
implement the model, learned geometry math, training loop, evaluation runtime,
AXC-out decoder, or interpreter runtime.

## Entry Points

Python:

```python
from hcaps.axt import AxtBundle, AxtCompileConfig, AxtDataset, compile_axt
from hcaps.axt.batch import AxtBatchCollator

compile_axt(AxtCompileConfig(...))
bundle = AxtBundle("bundle.axt")
dataset = AxtDataset("bundle.axt")
batch = AxtBatchCollator()([dataset[0]])
```

CLI:

```bash
axiom axt compile --input examples/axf/v0_1/minimal_dataset.axp --output artifacts/axt/minimal.axt --config configs/axt/compile_smoke.yaml --allow-all-without-split
axiom axt inspect artifacts/axt/minimal.axt
axiom axt validate artifacts/axt/minimal.axt
axiom axt tensor artifacts/axt/minimal.axt --group text_projection
axiom axt compare artifacts/axt/a.axt artifacts/axt/b.axt
```

## Bundle Contract

Every P3 bundle contains:

```text
axiom.json
tensors/*.safetensors
registries/field_registry.json
registries/vocabulary_registry.json
manifests/compile_config.json
manifests/tensor_manifest.json
manifests/hashes.json
manifests/source_index.json
reports/validation_report.json
reports/mask_report.json
reports/missing_target_report.json
```

P4 should read AXT through `AxtBundle` or `AxtDataset`, not by parsing AXC.

## Tensor Groups

Available groups:

```text
ids
claim
temporal
lateral_context
provider_context
epistemic_state
provenance
relations
relation_neighborhoods
negative_samples
geometry_observables
text_projection
targets
availability_masks
loss_masks
split_masks
metadata
```

## Predictor-Visible Inputs

P4 structured input should consume:

```text
ids
claim
temporal
lateral_context
provider_context
epistemic_state
provenance
relations
relation_neighborhoods
negative_samples
geometry_observables
text_projection
metadata
```

Text projection remains secondary. P4 must not treat `text_projection` as the
native substrate.

## Targets And Masks

AXC-out-aligned targets are in `targets`:

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

Availability masks:

```text
available_relation_targets
available_provenance_targets
available_epistemic_targets
available_future_targets
available_geometry_targets
available_text_targets
available_evaluation_references
```

Loss masks:

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

Missing targets are masked. They are not negative labels.

## Provider Context

Provider tensors include:

```text
provider_id_idx
provider_family_idx
provider_mode_idx
provider_config_hash_ref
prompt_template_idx
cascade_stage_idx
escalation_reason_idx
merge_strategy_idx
merge_confidence
disagreement_score
provider_vote_distribution
cache_replay_flag
provider_context_mask
```

Provider raw outputs and secrets are not compiled into tensors.

## Negative Samples

Negative tensors include:

```text
negative_type_idx
negative_source_idx
negative_target_idx
negative_hardness
negative_sampling_method_idx
negative_offsets
negative_mask
negative_loss_eligibility_mask
```

P3 normalizes P1 pool names to the P2 vocabulary where needed.

## Relation Neighborhoods

Pairwise neighborhoods are implemented now. Hypergraph fields are reserved:

```text
neighbor_node_values
neighbor_edge_type_values
edge_feature_values
node_offsets
edge_offsets
neighborhood_masks
hyperedge_id
hyperedge_type
hyperedge_incidence_values
hyperedge_offsets
hypergraph_reserved
```

P4 should keep the n-ary/hypergraph path open.

## Geometry Slots

Geometry tensors include gauge-invariant slots only:

```text
geometry_enabled
context_node_ids
context_edge_ids
context_loop_ids
transport_path_ids
curvature_observable_placeholder
holonomy_norm_placeholder
trace_summary_placeholder
spectrum_summary_placeholder
context_lability
geometry_loss_mask
geometry_ablation_mask
```

No raw connection matrices are semantic AXT fields.

## Text Projection

Text projection tensors include:

```text
flat_text_input_ids
flat_text_attention_mask
structured_text_input_ids
structured_text_attention_mask
capsule_text_input_ids
capsule_text_attention_mask
text_input_ids
text_attention_mask
text_labels
text_loss_mask
text_projection_targets
text_render_mode
special_token_ids
```

Text projection excludes future/evaluation-only fields and remains secondary.

## ID Lookup

Use `manifests/source_index.json` for:

```text
index_to_capsule_id
index_to_claim_family_id
index_to_claim_state_id
index_to_context_id
index_to_source_id
index_to_provider_trace_id
```

The same file stores field visibility, vocabulary snapshots, rendered-text audit
sidecars, and record metadata.

## P4 Boundary

P4 should implement:

```text
AxtDataset/AxtBatch -> structured encoder -> relation/hypergraph conditioning -> core/router -> structured decoder -> AXC-out + text projection
```

P4 should not implement:

```text
AXC parser as model input
training loop
learned geometry math beyond integration hooks
evaluation verdict
hidden interpreter oracle
```
