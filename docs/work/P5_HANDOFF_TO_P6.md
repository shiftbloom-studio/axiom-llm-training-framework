# P5 Handoff To P6

Status: P5 implementation-complete handoff for training, ablation, scoring, and verdict runtime.

Exact next action:

```text
Implement P6: Training, Experiments, Verdict & Operator Runtime.
```

P5 implemented a torch-native learned geometry module. It did not train it, evaluate it, tune regularizer weights, or claim that geometry improves performance.

## Public API

```python
from hcaps.geometry import (
    GeometryConfig,
    GeometryMode,
    ClaimFieldGraphBatch,
    LearnedConnection,
    LearnedGeometryModule,
    NoGeometryModule,
    NonGeometricContextMixer,
    P4GeometryProvider,
)
```

Main module:

```python
output = geometry_module(
    node_states=node_states,
    graph_batch=graph_batch,
    context_features=context_features,
    temporal_features=temporal_features,
    masks=masks,
)
```

P4 hook:

```python
model = AxiomStructuredModel(
    model_config,
    geometry_provider=P4GeometryProvider(geometry_config, model_dim=model_config.model_dim),
)
```

## Config Defaults

Config files:

- `configs/geometry/geometry_off.yaml`
- `configs/geometry/geometry_learned_smoke.yaml`
- `configs/geometry/geometry_precomputed_smoke.yaml`
- `configs/geometry/geometry_reference_smoke.yaml`

Default smoke geometry uses:

```text
mode = learned
fiber_dim = 16
transport_operator = matrix_exp
max_loop_length = 4
max_loops_per_batch = 128
regularizer weights = 0.0
export.include_raw_matrices = false
```

P6 decides regularizer weights.

## Graph Batch Schema

`ClaimFieldGraphBatch` contains:

```text
node_ids
node_type_ids
node_to_claim_state
relation_edge_index
relation_type_ids
relation_confidence
context_node_ids
context_edge_index
context_transition_type_ids
path_edge_index
path_mask
loop_path_index
loop_mask
hyperedge_incidence
temporal_features
lateral_context_features
provider_features
masks
```

Time is represented in `temporal_features`; it is not used as lateral context identity.

## Required AXT Inputs

P5 graph construction consumes P3/P4 groups:

```text
ids
relations
relation_neighborhoods
lateral_context
provider_context
temporal
geometry_observables
negative_samples
```

It does not parse AXC or source documents.

## GeometryOutput Fields

```text
mode
conditioning_features
observables
regularizer_terms
diagnostics
axc_out_fields
```

`conditioning_features` are torch tensors. `axc_out_fields` are JSON-safe gauge-invariant fields.

## AXC-out Geometry Export

Allowed exported fields:

```text
geometry.enabled
geometry.mode
geometry.gauge_policy
geometry.curvature_score
geometry.holonomy_norm
geometry.trace_normalized
geometry.spectrum_summary
geometry.path_consistency_score
geometry.context_lability_score
geometry.loop_count
```

Raw matrices, raw coefficients, provider secrets, and truth-like fields are not exported.

## Regularizer Terms

P5 returns unweighted differentiable terms:

```text
connection_norm
curvature_energy
path_consistency
context_smoothness
transport_identity_bias
non_degenerate_usage
```

P6 owns weights and schedules.

## Controls

Available controls:

- geometry off via `NoGeometryModule`
- parameter-matched non-geometric baseline via `NonGeometricContextMixer`
- context shuffle helper
- provider shuffle helper
- degree-preserving rewire compatibility helper

## Reference Mode

`hcaps.geometry.reference` implements a numpy reference path for small graph sanity checks. It is off-loop and not used for in-training learned geometry.

## Known Limitations

- P5 loop sampling is bounded and deterministic; it does not enumerate all cycles.
- Hyperedge incidence is preserved and pairwise expansion is available; native hypergraph message passing is reserved.
- Geometry is initialized near identity, so smoke fixtures may show zero curvature before training.
- P5 does not run experiments or provide evidence that geometry helps.

## Recommended P6 Arms

P6 should support at least:

```text
structured_native_no_geometry
structured_native_non_geometric_context_mixer
structured_native_learned_geometry
structured_native_learned_geometry_context_shuffle
structured_native_learned_geometry_provider_shuffle
structured_native_learned_geometry_degree_rewire
structured_native_learned_geometry_popularity_control
```
