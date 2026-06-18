# PLAN 4 / 6 — Structured-Native Model Stack

**Project:** Axiom — structured-native LLM training framework  
**Program:** P4 — Structured-Native Model Stack  
**Status:** implementation plan, ready after P3 contract/compiler handoff  
**Language:** Python 3.14  
**Primary framework:** PyTorch  
**Roadmap position:** P1 → P2 → P3 → **P4** → P6, with P5 developed in parallel after P3  
**Commit target:** `feat: add structured-native model stack`

---

## 0. Executive Summary

P4 builds Axiom's first real **structured-native LLM model stack**.

Axiom is still an LLM training framework, but it is no longer a normal token-in/token-out causal language model wrapper. Its primary model boundary is structured:

```text
AXT structured batch
  -> structured encoder
  -> relation / context / provider / provenance conditioning
  -> full-complexity core
  -> epistemic router
  -> structured decoder
  -> AXC-out raw emission
  -> text projection head
```

Text remains mandatory, but only as a **projection and comparison interface**. The model must remain compatible with language-model evaluation through a text-projection head, but text must not become the native substrate again.

P4 does **not** implement the learned geometry mathematics in full; that belongs to P5. P4 must, however, include stable geometry integration hooks, a geometry-off path, and shape-compatible placeholders so that P5 can plug into the core without reworking the model architecture.

P4 does **not** implement the training loop or final losses; that belongs to P6. P4 must emit loss-ready tensors, logits, masks, diagnostics, and structured output objects that P6 can train against.

---

## 1. Governing Concept

### 1.1 Axiom is structured-native, but LLM-compatible

The v1 target is a structured-native model with a text-projection head. It is not a stock Hugging Face causal LM.

The model must support:

```text
structured-native path:
  AXT -> structured latent state -> AXC-out

language-model-compatible path:
  same structured latent state -> text_projection_logits
```

Both paths are required.

Removing text projection would turn Axiom into a non-LLM graph model. Reducing the model to text projection only would collapse Axiom back into ordinary LLM training. P4 must preserve both.

### 1.2 Full complexity must persist through the model

The model must not encode rich structure only to flatten it into a token stream before the core.

The core design principle is:

```text
structured encode-in -> full-complexity core -> structured decode-out
```

In other words:

- claim identity remains visible;
- epistemic state remains visible;
- time remains separate from lateral context;
- provenance remains visible;
- relation neighborhoods remain visible;
- provider context remains visible;
- geometry hooks remain visible;
- AXC-out remains structured;
- text is projected from structure, not the only output.

### 1.3 Time is not lateral context

P4 must preserve the separation defined by P2/P3:

```text
time = temporal drift axis of the claim-state
lateral context = provider/source/community/method/domain/framing/etc.
```

Do not merge temporal features into a generic context embedding. The encoder must contain distinct temporal and lateral-context components.

### 1.4 Geometry is integrated but not completed here

P4 must provide geometry slots and integration points, but P5 owns the actual learned geometry implementation.

Allowed in P4:

- geometry feature adapter;
- geometry-off path;
- geometry placeholder tensors;
- geometry diagnostic pass-through;
- geometry integration into router/core if present.

Not allowed in P4:

- claiming HKR is proven;
- implementing final holonomy/curvature math as the primary task;
- raw connection matrices as semantic outputs;
- non-ablatable geometry assumptions.

### 1.5 Interpreter must not become an oracle

P4 may define structural decoder outputs and basic validation hooks, but the interpreter must be bounded.

The output layers must distinguish:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

P4 primarily produces **raw structured emissions** and text logits. Validation/interpreter utilities may be scaffolds, but they must not repair model errors using hidden source truth, future data, or external heuristics that would inflate model performance.

---

## 2. Preconditions

P4 starts after P3 provides a stable runtime interface. The P4 agent must first inspect the actual P3 outputs and bind to them.

Expected P3 artifacts include some equivalent of:

```text
src/hcaps/axt/
  runtime.py
  manifest.py
  specs.py
  registry.py
  tensor_bundle.py
  dataset.py
  batch.py

spec/AXT_TENSOR_BUNDLE.md
spec/AXC_OUT_SCHEMA.md
spec/FIELD_REGISTRY.md or equivalent
spec/VOCABULARY_REGISTRY.md or equivalent
```

Expected runtime data shape:

```text
AxtRuntimeBatch or equivalent
  claim_identity
  claim_state
  epistemic_state
  temporal_scope
  lateral_context
  provider_context
  provenance
  relations
  relation_neighborhoods
  negative_samples
  geometry_observable_slots
  text_projection
  targets_axc_out
  targets_text_projection
  loss_masks
  availability_masks
  metadata_index
```

If names differ, adapt to the actual P3 interface. Do not invent a parallel runtime format.

If P3 is missing or incomplete, stop and write a clear blocker report rather than building a second incompatible model input path.

---

## 3. P4 Objective

Build the first real Axiom structured-native model stack:

```text
AxtRuntimeBatch
  -> StructuredInputAdapter
  -> TypedFieldEmbeddings
  -> StructuredEncoder
  -> RelationNeighborhoodConditioner
  -> FullComplexityCore
  -> EpistemicRouter
  -> StructuredDecoder
  -> AXCOutEmission
  -> TextProjectionHead
```

The implementation must be small enough for smoke training on CPU/GPU, but real enough that P6 can later train it as an actual model.

Target model sizes should be configurable:

```text
smoke: 0.5M–2M parameters
small: 5M–15M parameters
v1_tiny_research: 10M–50M parameters
```

Exact numbers may vary, but the model must expose parameter counting and config-driven scaling.

---

## 4. Non-Goals

P4 must not implement:

- full training loop;
- optimizer/scheduler/checkpoint runtime;
- final multi-objective loss system;
- final learned geometry mathematics;
- final evaluation verdict;
- provider ingress;
- AXT compiler;
- corpus construction;
- external LLM calls;
- benchmark claims;
- any claim that Axiom works better.

P4 may include tests, smoke fixtures, and tiny forward-pass examples as part of implementation quality. These are not separate roadmap steps.

---

## 5. Package Layout

Create or update:

```text
src/hcaps/model/
  __init__.py
  config.py
  types.py
  batch.py
  embeddings.py
  input_adapter.py
  encoder.py
  relation_conditioning.py
  provenance_conditioning.py
  temporal.py
  context.py
  provider.py
  core.py
  router.py
  decoder.py
  heads.py
  axc_out.py
  text_projection.py
  geometry_hooks.py
  interpreter.py
  diagnostics.py
  factory.py
  parameter_count.py
  serialization.py

configs/model/
  structured_native_smoke.yaml
  structured_native_small.yaml
  structured_native_research_tiny.yaml

docs/MODEL_STACK.md

tests/model/
  test_model_config.py
  test_input_adapter.py
  test_structured_encoder.py
  test_relation_conditioning.py
  test_epistemic_router.py
  test_structured_decoder.py
  test_text_projection.py
  test_geometry_hooks.py
  test_model_forward_smoke.py
  test_model_ablation_modes.py
  test_model_serialization.py
```

If the repository uses a different config or test layout, follow the existing convention but preserve the conceptual boundaries.

---

## 6. Core Public API

P4 should expose a clean public model API.

### 6.1 Factory

```python
from hcaps.model import AxiomStructuredModel, AxiomModelConfig

config = AxiomModelConfig.from_yaml("configs/model/structured_native_smoke.yaml")
model = AxiomStructuredModel(config)
output = model(batch)
```

### 6.2 Model input

The model should accept the P3 runtime batch object if available.

If P3 emits dictionaries, provide a typed adapter:

```python
structured_inputs = StructuredInputAdapter(config).from_axt_batch(batch)
output = model(structured_inputs)
```

Do not require the model to parse raw AXC records.

P4 consumes compiled AXT runtime batches, not source documents and not AXC strings.

### 6.3 Model output

The forward pass must return a typed object similar to:

```python
AxiomModelOutput(
    latent_state=...,
    raw_axc_out=...,
    text_projection_logits=...,
    auxiliary_logits=...,
    router_diagnostics=...,
    geometry_diagnostics=...,
    masks=...,
    metadata=...,
)
```

The output must be loss-ready for P6 but should not compute final losses in P4.

---

## 7. Model Configuration

Implement `AxiomModelConfig` as a strict Pydantic model or dataclass with strong validation.

Required fields:

```text
model_dim
slot_dim
num_layers
num_heads
dropout
activation
max_claim_slots
max_relation_neighbors
max_provenance_sources
max_negative_samples
text_vocab_size
axc_out_vocab_sizes
use_text_projection
use_epistemic_router
use_relation_conditioning
use_provenance_conditioning
use_provider_context
use_geometry_features
geometry_mode
router_mode
core_type
parameter_budget_hint
```

Recommended config sections:

```yaml
model:
  model_dim: 256
  slot_dim: 256
  num_layers: 4
  num_heads: 4
  dropout: 0.1

inputs:
  use_claim_identity: true
  use_epistemic_state: true
  use_temporal_scope: true
  use_lateral_context: true
  use_provider_context: true
  use_provenance: true
  use_relations: true
  use_relation_neighborhoods: true
  use_negative_samples: true
  use_geometry_features: true

router:
  enabled: true
  num_experts: 4
  router_hidden_dim: 128
  load_balancing: false

outputs:
  emit_axc_out: true
  emit_text_projection: true
  emit_auxiliary_heads: true

ablations:
  no_provenance: false
  no_relations: false
  no_context: false
  no_provider_context: false
  no_side_channels: false
  geometry_off: false
  router_off: false
  text_only: false
```

The config must support ablation modes without code changes.

---

## 8. Input Adapter

Implement `StructuredInputAdapter`.

Purpose:

```text
P3 runtime batch -> normalized model input object
```

The adapter must:

- accept the actual P3 batch object or dictionary;
- map tensor groups into typed fields;
- enforce required/missing input behavior;
- preserve masks;
- preserve metadata IDs;
- preserve temporal vs lateral-context separation;
- not parse AXC manually;
- not perform source-level feature extraction;
- not call providers;
- not infer hidden truth.

Output type should be something like:

```python
AxiomModelInput(
    claim_identity=...,
    claim_state=...,
    epistemic_state=...,
    temporal_scope=...,
    lateral_context=...,
    provider_context=...,
    provenance=...,
    relations=...,
    relation_neighborhoods=...,
    negative_samples=...,
    geometry_features=...,
    text_projection_input=...,
    loss_masks=...,
    availability_masks=...,
    metadata=...,
)
```

If P3 provides already typed objects, the adapter may mostly validate and normalize them.

---

## 9. Typed Field Embeddings

P4 must not concatenate everything into one flat feature vector at the boundary.

Implement typed field embedding modules:

```text
ClaimIdentityEmbedding
ClaimStateEmbedding
EpistemicStateEmbedding
TemporalEmbedding
LateralContextEmbedding
ProviderContextEmbedding
ProvenanceEmbedding
RelationEmbedding
RelationNeighborhoodEmbedding
NegativeSampleEmbedding
GeometryFeatureEmbedding
TextProjectionInputEmbedding
```

Each module maps its structured group to one or more **typed slots**.

A typed slot should carry:

```text
slot_tensor
slot_type_id
slot_mask
source_group
optional metadata reference
```

The structured encoder then consumes a set/sequence of typed slots.

### 9.1 Claim identity embedding

Should support at least:

- claim family ID embedding;
- claim state ID embedding if available;
- canonical claim text embedding through text projection input if provided;
- field-registry categorical IDs.

### 9.2 Epistemic embedding

Should encode:

- O / ontology compatibility;
- E / evidential anchoring;
- T / transformation pressure;
- R / independent redundancy;
- uncertainty;
- stability/status if present;
- availability masks.

The model must not treat missing labels as zero evidence unless P2 explicitly defines that behavior. Missingness must be represented through masks.

### 9.3 Temporal embedding

Should encode temporal features separately from lateral context:

- valid-as-of;
- observed-at;
- source-publication deltas;
- cutoff deltas;
- future-target masks;
- temporal split features.

Implementation may use normalized scalar time features and/or learned bucket embeddings.

### 9.4 Lateral-context embedding

Should encode non-temporal context:

- field/domain;
- subfield;
- community;
- method context;
- source/venue cluster;
- language/register;
- extractor/provider slant if P2/P3 routes it here.

Provider context may have its own module but remains lateral context semantically.

### 9.5 Provider-context embedding

Should encode P1/P2 provider signals:

- provider ID;
- provider type;
- cascade stage;
- escalation reason;
- merge confidence;
- disagreement score;
- provider vote distribution if present;
- cached/replayed flag if present.

Provider context must be ablatable.

### 9.6 Provenance embedding

Should encode:

- source IDs;
- evidence span IDs or indices;
- source type;
- license class if provided;
- source timestamp features;
- provenance confidence;
- source count.

It should support variable numbers of sources through masks.

### 9.7 Relation and neighborhood embedding

Should encode:

- relation type;
- source/target claim family references;
- relation confidence;
- neighbor role;
- edge direction;
- relation evidence pointer if available;
- hard-negative type if applicable.

Relation-neighborhood conditioning is first-class and must not be dropped.

### 9.8 Geometry feature embedding

P4 should accept geometry feature tensors from P3/P5 slots if present:

- curvature score placeholder;
- holonomy norm placeholder;
- trace summary placeholder;
- spectrum summary placeholder;
- transport inconsistency placeholder;
- context lability placeholder.

If geometry is absent/off, the module must output stable zero/neutral slots and masks.

---

## 10. Structured Encoder

Implement a structured encoder that consumes typed slots and produces a latent representation.

Recommended first version:

```text
typed field slots
  -> type embeddings
  -> slot projection
  -> slot self-attention / set transformer blocks
  -> relation-conditioned refinement
  -> pooled global claim-state representation
```

This should not be a standard token-only transformer. It may use attention, but over structured slots and relation/neighborhood representations.

### 10.1 Encoder outputs

The encoder should emit:

```text
slot_states
claim_state_vector
epistemic_vector
relation_context_vector
provenance_context_vector
temporal_vector
lateral_context_vector
provider_context_vector
geometry_context_vector
encoder_diagnostics
```

### 10.2 Slot type preservation

Slot type IDs must remain available to later modules.

Do not collapse everything irreversibly into one vector before relation/core/router stages.

---

## 11. Relation / Hypergraph Conditioning

Implement `RelationNeighborhoodConditioner`.

Purpose:

```text
relation tensors + neighbor tensors -> relation-conditioned slot states
```

The first implementation may be simple message passing over fixed relation-neighborhood arrays.

It must support:

- relation type embeddings;
- direction embeddings;
- neighbor claim embeddings;
- relation confidence features;
- hard-negative type embeddings;
- relation masks;
- no-relation ablation;
- open path for n-ary/hyperedge relations.

Recommended structure:

```text
RelationEdgeEncoder
RelationNeighborhoodAggregator
RelationConditionedAttention
HyperedgePlaceholderAdapter
```

### 11.1 N-ary path must remain open

Even if P4 implements only binary relation arrays initially, the code and docs must not hard-code a binary-only worldview.

Use names like:

```text
relation_incidence
edge_or_hyperedge_id
participant_role
arity
```

where feasible.

If n-ary support is not implemented, create a clean extension interface and tests that confirm binary behavior does not block future n-ary tensors.

---

## 12. Provenance Conditioning

Implement `ProvenanceConditioner`.

Purpose:

```text
source/evidence span tensors -> provenance-aware claim-state representation
```

It should support:

- source embedding aggregation;
- evidence span embedding aggregation;
- temporal compatibility features;
- provenance count / diversity features;
- provenance masking;
- no-provenance ablation.

It must not use provenance as a hidden oracle.

Provenance features can condition the model, but if an evaluation arm masks provenance, the model must not recover it through leaked fields.

---

## 13. Full-Complexity Core

Implement `FullComplexityCore`.

Purpose:

```text
structured encoded slots -> reasoning latent state
```

This core is the heart of the structured-native model stack.

Recommended first implementation:

```text
N layers of ClaimFieldBlock

ClaimFieldBlock:
  typed-slot self-attention
  cross-attention to relation/provenance/neighborhood states
  gated MLP
  residual + norm
  optional geometry feature conditioning
  optional router modulation
```

Alternative simple implementation allowed if kept structured:

```text
slot mixer + relation-conditioned attention + gated feedforward
```

Do not replace the core with a plain text transformer over serialized capsule text.

### 13.1 Core outputs

The core should return:

```text
core_slot_states
core_claim_state_vector
core_relation_state
core_provenance_state
core_context_state
core_router_state
core_diagnostics
```

### 13.2 Geometry integration hooks

The core must expose hook points for P5:

```python
geometry_context = geometry_provider(input, slot_states, relation_graph)
slot_states = core_block(..., geometry_context=geometry_context)
```

If geometry is off, the same code path should run with neutral geometry context.

---

## 14. Epistemic Router

Implement `EpistemicRouter` as a real first-class module.

The router may be simple in v1, but it must exist and be ablatable.

Purpose:

```text
epistemic/context/uncertainty features -> routing/gating/loss-steering signals
```

Possible outputs:

```text
expert_weights
head_weights
loss_term_weights_for_P6
routing_logits
uncertainty_gate
revision_gate
established_gate
router_diagnostics
```

The router should condition on:

- epistemic state;
- uncertainty;
- transformation pressure;
- redundancy;
- relation density;
- provenance density;
- temporal features;
- provider disagreement;
- optional geometry context.

### 14.1 Router modes

Support config modes:

```text
router_off
scalar_gate
head_gate
expert_gate
loss_steering_only
```

The first implementation can use `scalar_gate` or `head_gate`, but the interfaces should allow later MoE/expert gating.

### 14.2 Router must not decide truth

The router must not output `truth`, `correctness`, or binary factuality.

It may output:

```text
established_gate
revision_gate
uncertainty_gate
relation_head_weight
provenance_head_weight
future_head_weight
geometry_head_weight
```

These are computational routing signals, not truth labels.

---

## 15. Structured Decoder

Implement `StructuredDecoder`.

Purpose:

```text
core latent state -> AXC-out raw emission tensors
```

The decoder must emit structured predictions corresponding to P2 AXC-out targets.

At minimum:

```text
claim_state_head
relation_head
provenance_head
epistemic_head
stability_head
future_summary_head_or_placeholder
geometry_observable_head_or_placeholder
```

### 15.1 AXC-out raw emission

Define an object like:

```python
AXCOutRawEmission(
    claim_state_logits=...,
    relation_type_logits=...,
    relation_target_logits=...,
    provenance_source_logits=...,
    evidence_span_logits=...,
    epistemic_values=...,
    uncertainty_values=...,
    stability_logits=...,
    future_summary_latent=...,
    geometry_observable_values=...,
    masks=...,
    metadata=...,
)
```

This is not yet a fully interpreted AXC-out document. It is the model's raw structured output.

### 15.2 Validation hook

Implement a lightweight validation hook that checks shapes, masks, and allowed output names.

Do not implement hidden repair logic.

The hook may return:

```text
valid_shape
valid_mask_alignment
invalid_missing_required_head
invalid_forbidden_truth_field_name
```

---

## 16. Text Projection Head

Implement `TextProjectionHead`.

Purpose:

```text
core latent state + optional text projection input -> token logits
```

The head must support:

- vocabulary size from P3 vocabulary registry;
- optional teacher-forced text input if P3 provides it;
- token logits for P6 text-projection loss;
- text-only baseline compatibility;
- masking/padding compatibility.

### 16.1 Text projection is mandatory

Do not treat text projection as optional at the project level. It may be disabled in ablations, but the v1 model stack must implement it.

This preserves LLM comparability.

### 16.2 Text projection is secondary

The text projection head should not become the only model output.

The forward output must include both:

```text
raw_axc_out
text_projection_logits
```

---

## 17. Auxiliary Heads

Implement auxiliary heads in `heads.py`.

Required head classes:

```text
RelationPredictionHead
ProvenanceRecoveryHead
EpistemicRegressionHead
StabilityPredictionHead
FutureSummaryLatentHead
GeometryObservableHead
TextProjectionHead
```

The heads should be shape-compatible with P2/P3 target tensors and masks.

P6 will define losses. P4 only emits predictions.

---

## 18. Interpreter Boundary

Implement a minimal `AXCOutInterpreterBoundary`, not a full oracle interpreter.

Purpose:

```text
raw emission -> optional validated skeleton/projection for debugging
```

Allowed behavior:

- convert logits to top-k candidate fields;
- attach model confidence;
- validate enum names;
- validate no forbidden truth fields;
- preserve raw outputs;
- mark fields as model-emitted.

Disallowed behavior:

- adding missing provenance from input graph unless explicitly emitted;
- repairing relation targets using known gold references;
- injecting future labels;
- using external provider calls;
- turning uncertain outputs into established claims;
- hiding invalid raw emissions.

The interpreter boundary should produce a debug artifact, not a scored final product.

P6 will decide scoring rules.

---

## 19. Geometry Hooks

Implement `geometry_hooks.py`.

Required interfaces:

```python
class GeometryProviderProtocol(Protocol):
    def forward(...): ...

class NullGeometryProvider(nn.Module):
    ...

class GeometryContext:
    observables: Tensor | None
    conditioning: Tensor | None
    diagnostics: dict[str, Any]
```

P4 must support:

```text
geometry_off
geometry_features_from_axt
geometry_provider_injected
geometry_diagnostics_pass_through
```

P5 will replace or extend these hooks with learned geometry.

Do not implement final P5 connection/holonomy/curvature math here unless it is only a stub/reference shape.

---

## 20. Ablation Modes

P4 must support config-driven ablations.

Required ablations:

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

Ablation must be explicit, not achieved by deleting fields randomly.

Each ablation should be visible in diagnostics.

Example:

```python
output.diagnostics["ablations"] = {
    "no_provenance": True,
    "geometry_off": True,
}
```

---

## 21. Diagnostics

Implement structured diagnostics.

Diagnostics are required for P6 and debugging.

Include:

```text
parameter_count
enabled_modules
active_ablation_modes
input_shape_summary
slot_count_by_type
mask_summary
router_summary
geometry_summary
output_shape_summary
text_projection_shape
axc_out_head_shapes
```

Diagnostics must not include private secrets, provider keys, or raw API credentials.

---

## 22. Serialization

Implement model save/load utilities.

Required:

```python
save_model_checkpoint(path, model, config, metadata)
load_model_checkpoint(path) -> model, config, metadata
```

Use standard PyTorch `state_dict` for now.

Metadata should include:

```text
model_config
parameter_count
axiom_version
p4_model_schema_version
created_at
compatible_axt_spec_version
compatible_axc_out_spec_version
```

P6 may later replace this with full training checkpointing.

---

## 23. Config Files

Add three configs.

### 23.1 `structured_native_smoke.yaml`

Purpose: CPU-friendly shape/forward tests.

Approximate:

```yaml
model_dim: 64
num_layers: 2
num_heads: 2
router:
  enabled: true
outputs:
  text_projection: true
  axc_out: true
```

### 23.2 `structured_native_small.yaml`

Purpose: local development model.

Approximate:

```yaml
model_dim: 128
num_layers: 4
num_heads: 4
router:
  enabled: true
```

### 23.3 `structured_native_research_tiny.yaml`

Purpose: first P6 research pilot.

Approximate:

```yaml
model_dim: 256
num_layers: 6
num_heads: 8
router:
  enabled: true
  mode: head_gate
```

Actual values may be adjusted for repository conventions.

---

## 24. Documentation

Add `docs/MODEL_STACK.md`.

It must explain:

- Axiom remains a structured-native LLM training framework;
- why the model is not stock CausalLM;
- how AXT enters the model;
- structured encoder;
- relation-neighborhood conditioning;
- epistemic router;
- full-complexity core;
- structured decoder;
- AXC-out raw emission;
- text projection;
- geometry hooks;
- ablation modes;
- P4 vs P5 vs P6 boundaries.

Update README only if necessary to mention that P4 introduces the structured-native model stack.

Do not claim model performance.

---

## 25. CLI and Developer Surface

P4 does not need a full operator CLI, but it should expose minimal developer commands if the repository CLI pattern supports it.

Optional commands:

```bash
axiom model config-summary configs/model/structured_native_smoke.yaml
axiom model count-params configs/model/structured_native_smoke.yaml
axiom model smoke-forward <axt_bundle_or_fixture> --config configs/model/structured_native_smoke.yaml
```

If adding CLI commands would introduce too much coupling before P6, skip CLI and provide Python examples/tests instead.

Do not build training commands in P4.

---

## 26. Tests and Quality Gates

Testing is part of implementation, not a separate roadmap step.

Required tests:

### 26.1 Config tests

- valid smoke config loads;
- invalid dimensions fail clearly;
- ablation flags validate;
- geometry mode validates;
- parameter budget is reported.

### 26.2 Input adapter tests

- adapter consumes P3 runtime fixture;
- required groups are mapped correctly;
- masks are preserved;
- time and lateral context remain separate;
- missing optional geometry produces neutral geometry context.

### 26.3 Embedding tests

- each field embedding returns expected shapes;
- missing targets/masks are handled;
- provider context is ablatable;
- relation neighborhood embeddings support masks.

### 26.4 Encoder/core tests

- structured encoder forward pass works;
- typed slots remain tracked;
- core preserves batch dimension;
- ablations do not crash;
- no token-only bypass becomes the default.

### 26.5 Router tests

- router off path works;
- router on path returns diagnostics;
- router outputs are finite;
- router does not emit forbidden truth labels.

### 26.6 Decoder/head tests

- AXC-out heads emit required tensors;
- text projection emits `[batch, seq, vocab]` or documented equivalent;
- auxiliary heads align with masks;
- invalid output names are rejected.

### 26.7 Geometry hook tests

- null geometry provider works;
- geometry feature path works if features are present;
- geometry_off ablation is shape-compatible;
- raw geometry matrices are not emitted as semantic output.

### 26.8 Full forward smoke test

Using a small P3 AXT fixture:

```python
batch = load_fixture_axt_batch(...)
model = AxiomStructuredModel(smoke_config)
output = model(batch)
```

Assert:

```text
raw_axc_out exists
text_projection_logits exists
router diagnostics exist if enabled
geometry diagnostics exist
no NaNs
batch dimensions preserved
```

### 26.9 Serialization tests

- save/load config and state dict;
- loaded model forward output shape matches original;
- metadata includes compatible AXT and AXC-out spec versions.

Run:

```bash
ruff check .
ruff format .
mypy src
pytest
```

If full PyTorch dependency is not yet installed in CI, add optional markers or CI notes, but P4 must clearly declare torch as required for model stack tests.

---

## 27. Quality Bar

The P4 implementation must be:

- torch-native;
- typed;
- config-driven;
- ablatable;
- shape-tested;
- small-model runnable;
- compatible with P3 AXT runtime;
- compatible with P5 geometry integration;
- compatible with P6 training/losses;
- honest about non-goals;
- free of benchmark claims;
- free of truth labels.

Do not accept a P4 implementation that only serializes capsules to text and calls a standard CausalLM.

Do not accept a P4 implementation that produces AXC-out only through a postprocessor over text logits.

AXC-out must be a native structured emission path.

---

## 28. Definition of Done

P4 is complete when:

```text
[ ] src/hcaps/model package exists.
[ ] AxiomModelConfig exists and loads model configs.
[ ] StructuredInputAdapter consumes P3 runtime batches.
[ ] Typed field embeddings exist for all required groups.
[ ] Structured encoder exists.
[ ] Relation-neighborhood conditioning exists.
[ ] Provenance conditioning exists.
[ ] Temporal and lateral context encoders are separate.
[ ] Provider-context conditioning exists and is ablatable.
[ ] FullComplexityCore exists.
[ ] EpistemicRouter exists and is ablatable.
[ ] StructuredDecoder emits AXC-out raw emissions.
[ ] TextProjectionHead emits token logits.
[ ] Geometry hook interface exists with null provider.
[ ] Ablation modes are config-driven.
[ ] Model forward pass works on a P3 AXT fixture.
[ ] Model output includes raw_axc_out and text_projection_logits.
[ ] Diagnostics are emitted.
[ ] Save/load works.
[ ] Docs explain P4 boundaries.
[ ] Tests pass.
[ ] No performance claims are made.
```

---

## 29. P4 Handoff to P5

P4 must produce `docs/work/P4_HANDOFF_TO_P5.md` or equivalent.

It should document:

```text
geometry hook API
expected geometry input tensors
expected geometry conditioning tensors
where geometry enters the core
where geometry diagnostics are emitted
how geometry_off works
how context_shuffle can be tested later
how P5 can replace NullGeometryProvider
```

P5 should not need to refactor the entire model stack to integrate learned geometry.

---

## 30. P4 Handoff to P6

P4 must produce `docs/work/P4_HANDOFF_TO_P6.md` or equivalent.

It should document:

```text
model output object
raw AXC-out emission fields
text projection logits shape
auxiliary head outputs
router diagnostics
available ablation modes
expected target tensors from P3
expected loss masks from P3
unsupported target fields
known limitations
```

P6 should be able to write losses and training loops without guessing model semantics.

---

## 31. Agent Instructions

When implementing P4:

1. Do not reduce Axiom to a standard CausalLM.
2. Do not remove text projection.
3. Do not implement full training.
4. Do not implement final learned geometry math.
5. Do not call external LLMs.
6. Do not add truth labels.
7. Do not use AXC parsing inside the model.
8. Do not silently flatten time and lateral context.
9. Do not drop provider context.
10. Do not drop relation-neighborhood conditioning.
11. Do not make the interpreter an oracle.
12. Do not make geometry non-ablatable.
13. Keep the implementation small but real.

If a tradeoff arises, preserve the structured-native architecture even if the initial implementation is smaller.

---

## 32. Suggested Implementation Order Inside P4

This is not a separate validation plan. It is an internal build order for the agent.

```text
1. model config + typed input/output objects
2. input adapter for P3 AXT runtime batches
3. typed field embeddings
4. structured encoder
5. relation/provenance/provider/temporal/context conditioning
6. full-complexity core
7. epistemic router
8. structured decoder + AXC-out raw emission object
9. text projection head
10. geometry hook interface
11. ablation modes
12. diagnostics + parameter counting
13. serialization
14. docs + tests + quality gates
15. P4 handoff docs
```

Do not skip the output side. A structured encoder without AXC-out is not P4.

---

## 33. Expected Pull Request

Branch:

```text
feat/p4-structured-native-model-stack
```

Commit:

```text
feat: add structured-native model stack
```

PR title:

```text
P4: Structured-Native Model Stack
```

PR body must include:

```text
Summary
Implemented modules
Config files
AXT/P3 interface used
AXC-out emission fields
Text projection support
Router support
Geometry hook support
Ablation support
Tests run
Known limitations
P5 handoff
P6 handoff
```

No benchmark/performance claim should appear in the PR.

---

## 34. Final P4 Principle

P4 succeeds only if the model has two native output paths from the same structured latent state:

```text
structured latent state -> AXC-out raw emission
structured latent state -> text projection logits
```

If the AXC-out path is missing, the model is not structured-native.

If the text projection path is missing, the project is no longer an LLM training framework.

P4 must preserve both.
