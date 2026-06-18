# Axiom Model Stack

Status: P4 structured-native model stack implemented.

Axiom remains a structured-native LLM training framework. The model boundary is compiled AXT structure, not a plain token stream. Text projection is implemented and mandatory for comparability, but it is a secondary output path beside native AXC-out raw emissions.

```text
AxtBatch / AxtRecord
  -> StructuredInputAdapter
  -> typed field embeddings
  -> StructuredEncoder
  -> relation/provenance/context/provider conditioning
  -> FullComplexityCore
  -> EpistemicRouter
  -> StructuredDecoder
  -> AXC-out raw emission
  -> AXCOutInterpreterBoundary
  -> text projection logits
```

## Public API

```python
from hcaps.axt import AxtBatchCollator, AxtDataset
from hcaps.model import AxiomModelConfig, AxiomStructuredModel

config = AxiomModelConfig.from_yaml("configs/model/structured_native_smoke.yaml")
dataset = AxtDataset("artifacts/axt/minimal.axt")
batch = AxtBatchCollator()([dataset[0], dataset[1]])
model = AxiomStructuredModel(config)
output = model(batch)
```

The model consumes P3 runtime batches through `hcaps.axt`. It does not parse AXC strings or source documents.

## Why It Is Not Stock CausalLM

The stack uses attention, but attention runs over typed structured slots: claim identity, claim state, epistemic state, temporal scope, lateral context, provider context, provenance, relations, relation neighborhoods, negative samples, nullable/maskable geometry features, and text-projection input.

The model does not serialize capsules to text and hand them to a stock causal LM. The native structured path emits AXC-out raw tensors, while the text head emits token logits for text-projection losses and text-rendered comparisons.

## Outputs

`AxiomModelOutput` includes:

- `latent_state`
- `slot_states`
- `raw_axc_out`
- `text_projection_logits`
- `auxiliary_logits`
- `router_diagnostics`
- `geometry_diagnostics`
- `masks`
- `metadata`
- `diagnostics`

`raw_axc_out.raw_emission` includes:

- `claim_state_logits`
- `relation_type_logits`
- `relation_target_logits`
- `provenance_source_logits`
- `evidence_span_logits`
- `epistemic_status_logits`
- `epistemic_values`
- `uncertainty_values`
- `stability_logits`
- `future_summary_latent`
- `geometry_observable_values`

The interpreter boundary preserves `raw_emission`, reports `validated_axc_out`, and creates a debug `interpreted_projection` without hidden oracle repairs.

## Geometry Hooks

P4 provides `GeometryHook`, `NullGeometryProvider`, `AxtGeometryFeatureProvider`, and `GeometryProviderProtocol`.

Geometry enters the core as a conditioning vector in `FullComplexityCore`. The default smoke path is `geometry_off`. P5 can inject a learned provider through:

```python
model = AxiomStructuredModel(config, geometry_provider=my_geometry_provider)
```

P4 does not implement P5's learned connection, transport, holonomy, or curvature math. P4 only provides shape-compatible hooks and gauge-invariant observable slots.

## Ablations

Config-driven ablations include:

- `text_only`
- `structure_only_no_text_projection`
- `no_provenance`
- `no_relations`
- `no_context`
- `no_provider_context`
- `no_side_channels`
- `geometry_off`
- `router_off`
- `relation_neighborhood_off`

Ablations are reported in diagnostics.

## Configs And CLI

Model configs:

- `configs/model/structured_native_smoke.yaml`
- `configs/model/structured_native_small.yaml`
- `configs/model/structured_native_research_tiny.yaml`

Developer CLI:

```bash
axiom model config-summary configs/model/structured_native_smoke.yaml
axiom model count-params configs/model/structured_native_smoke.yaml
axiom model smoke-forward artifacts/axt/minimal.axt --config configs/model/structured_native_smoke.yaml
```

These commands do not train a model.

## Boundaries

P4 does not implement:

- learned P5 geometry math;
- training loops, optimizers, schedulers, or losses;
- experiment orchestration;
- benchmark results;
- external LLM calls;
- truth labels.

P5 now owns the learned geometry module through `hcaps.geometry` and the P4 `GeometryProviderProtocol`. P6 owns loss computation, training, experiment arms, and research verdicts.
