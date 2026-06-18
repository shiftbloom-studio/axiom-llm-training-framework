# Axiom Geometry Module

Status: P5 learned claim-field geometry implemented.

Axiom geometry is a trainable context-transport module over claim-field graphs. It is not a truth engine, proof of HKR, or a popularity counter. It exists to make a falsifiable question testable in P6: whether learned lateral context transport adds epistemic modeling value beyond structured input, auxiliary supervision, architecture changes, and ordinary graph mixing.

## Meaning

P5 treats lateral context transitions as learnable transport edges. A claim-state representation can move across contexts such as provider, source type, method, community, language, or domain. Loops through these contexts can produce holonomy-like signals. P5 reports only gauge-invariant observables.

Time remains separate from lateral context. Temporal features may condition geometry, but temporal cycles are not context loops.

## Public API

```python
from hcaps.geometry import GeometryConfig, LearnedGeometryModule
from hcaps.geometry.batching import graph_batch_from_axt_batch

config = GeometryConfig.from_yaml("configs/geometry/geometry_learned_smoke.yaml")
graph = graph_batch_from_axt_batch(batch, config)
module = LearnedGeometryModule(config, input_dim=64)
output = module(node_states=node_states, graph_batch=graph)
```

P4 integration uses:

```python
from hcaps.geometry import P4GeometryProvider

provider = P4GeometryProvider(config, model_dim=64)
model = AxiomStructuredModel(model_config, geometry_provider=provider)
```

## Modes

- `off`: neutral geometry-off ablation.
- `precomputed`: consumes existing gauge-invariant geometry observables without learned transport.
- `learned`: torch-native learned connection and transport path.
- `reference`: off-loop numpy reference path for small sanity checks.

## Observables

Exportable observables include:

- `curvature_score`
- `holonomy_norm`
- `trace_normalized`
- `spectrum_summary`
- `path_consistency_score`
- `context_lability_score`
- `loop_count`
- `mode`

Raw connection matrices, transport matrices, and basis coefficients are not semantic outputs.

## Controls

P5 provides:

- `NoGeometryModule`
- `NonGeometricContextMixer`
- deterministic context shuffle
- deterministic provider shuffle
- degree-preserving rewire compatibility helper

These support P6 ablations. P5 does not run experiments or decide verdicts.

## CLI

```bash
axiom geometry inspect-axt artifacts/axt/minimal.axt
axiom geometry sample-loops artifacts/axt/minimal.axt --output loops.json
axiom geometry compute-reference artifacts/axt/minimal.axt --output observables.json
axiom geometry smoke artifacts/axt/minimal.axt \
  --config configs/geometry/geometry_learned_smoke.yaml
```

These commands inspect and smoke-test geometry. They do not train models or produce performance claims.

## Boundaries

P5 does not implement:

- P6 training loops or loss weighting;
- benchmark evaluation;
- verdict reports;
- external LLM calls;
- truth labels;
- physical or mathematical proof claims.

P6 owns training, controlled arms, scoring, and verdict logic.
