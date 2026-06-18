# P4 Handoff To P5

Status: P4 implementation-complete handoff consumed by P5.

Current next action after P5:

```text
Implement P6: Training, Experiments, Verdict & Operator Runtime.
```

P4 implemented geometry hooks and a geometry-off path. It did not implement learned connection, transport, holonomy, curvature, gauge regularization, graph dynamics, training, or evaluation.

## Hook API

Main entry points:

```python
from hcaps.model import AxiomStructuredModel
from hcaps.model.geometry_hooks import GeometryContext, GeometryProviderProtocol

model = AxiomStructuredModel(config, geometry_provider=provider)
```

Required provider shape:

```python
provider.forward(
    model_input,
    slot_states,
    relation_graph=None,
) -> GeometryContext
```

`GeometryContext` fields:

```text
observables: Tensor | None
conditioning: Tensor | None
diagnostics: dict[str, Any]
```

`conditioning` is expected to be `[batch, model_dim]`. `observables` should contain gauge-invariant summaries only.

## Expected AXT Inputs

P4 passes the normalized `AxiomModelInput` to geometry providers. Relevant P3 groups:

```text
geometry_observables
relations
relation_neighborhoods
negative_samples
temporal
lateral_context
provider_context
```

P3 geometry fields are gauge-invariant placeholders/observables. P4 does not expose raw learned matrices as canonical semantic outputs.

## Core Integration Point

Geometry enters in `FullComplexityCore.forward(...)`:

```text
relation_state
provenance_state
temporal_state
lateral_state
provider_state
geometry_context.conditioning
  -> context_merge
  -> ClaimFieldBlock conditioning
```

P5 should be able to replace `NullGeometryProvider` without refactoring the encoder, decoder, router, or text projection head.

## Geometry-Off Path

`geometry_off` returns neutral zero conditioning through `NullGeometryProvider`.

Diagnostics include:

```text
enabled: False
mode: geometry_off
gauge_invariant_only: True
raw_gauge_matrices_emitted: False
```

## Context Shuffle Readiness

P4 keeps time, lateral context, provider context, relations, and provenance in separate groups and separate typed slots. P5/P6 can test context shuffles by replacing or permuting the relevant AXT groups before model forward, then comparing geometry-enabled and geometry-off behavior.

## Constraints For P5

P5 must preserve:

- no truth labels;
- no raw gauge matrices as semantic output;
- gauge-invariant observable reporting;
- geometry-off ablation;
- context-shuffle compatibility;
- AXC-out raw emission plus text projection from the same structured latent state.
