# Axiom v1 — Updated Implementation Roadmap

**Status:** consolidated six-program roadmap
**Target:** full structured-native LLM training framework v1
**Current next program:** P6 — Training, Experiments, Verdict & Operator Runtime
**Roadmap style:** all-or-nothing implementation programs; no standalone validation, test, cleanup, or documentation phases

---

## 1. Purpose

This roadmap is the current implementation source of truth for completing Axiom v1.

Axiom is a **structured-native LLM training framework**. It remains LLM-compatible through text projection, token baselines, text-rendered arms, and secondary text losses, but its primary substrate is not a flat token stream. Its primary substrate is structured claim-field data: claim identity, provenance, relations, epistemic state, temporal scope, lateral context, provider traces, negative pools, and optional gauge-invariant geometry.

The v1 implementation goal is not to build another wrapper around a standard token-in/token-out causal language model. The goal is to build a small but real structured-native model stack that can ingest Axiom structure, compute over it, emit structured AXC-out, and project to text for comparability.

The roadmap deliberately avoids separate validation or testing phases. Each implementation program must include its own tests, documentation updates, quality gates, examples, and handoff artifacts as part of the work. They are not separate roadmap steps.

---

## 2. Current State

The foundation work has already established the repository substrate:

- schema, storage, manifests, and temporal leakage guards;
- claim-field substrate builder;
- AXF / AXC / AXP format groundwork;
- conformance fixtures;
- initial training bridge;
- falsification artifact preparation;
- provider ingress and first corpus-generation layer.

P1 is reported as implementation-complete and provides:

- provider ingress layer with deterministic, OpenAI-compatible dry-run/cache-safe providers and Python-callable providers;
- cache and replay infrastructure;
- local/provider cascade and merge/disagreement traces;
- corpus builder with source registry and provider traces;
- synthetic views;
- epistemic proxies;
- negative pools;
- gold-reference hooks;
- AXP report enrichment;
- CLI commands;
- ML/software benchmark fixture configs and examples;
- optional/plugin-gated PDF reader path;
- pre-P1 documentation baseline alignment.

P1 is accepted as the substrate-ingress foundation for P2. P2 formalizes the contracts around P1 outputs. P3 implements the executable AXT compiler/runtime data interface over those contracts. P4 implements the first torch-native structured model stack over P3 AXT batches. P5 implements the learned claim-field geometry module and hands geometry controls/regularizers to P6.

---

## 3. Governing Principles

### 3.1 Structured-native LLM identity

Axiom is still an LLM training framework, but it is not a standard causal-LM framework. Its primary I/O is structured.

```text
AXC / AXP
  -> AXT structured tensors
  -> structured-native encoder/core/decoder
  -> AXC-out structured emission
  -> optional text projection
```

Text remains mandatory as a projection and comparison interface. Removing text projection would turn Axiom into a non-LLM graph system, which is not the v1 target.

### 3.2 Text is projection, not the system boundary

Text appears in:

- flat-text baselines;
- structured-text baselines;
- text projection head;
- human-readable reports;
- optional interpreted projections.

Text is not the native substrate. Capsule text and special-token streams are controlled projections, not the primary representation.

### 3.3 Claim-state, not truth

Axiom stores and predicts claim-states, not binary truth. AXF, AXT, and AXC-out must not introduce fields such as:

```text
truth
is_true
correct
is_correct
proven_true
factuality
```

Gold-reference hooks may exist only as evaluation references, not metaphysical truth labels.

### 3.4 Time is not lateral context

Time is the drift axis of the claim-state. Lateral context includes provider, source, community, method, field, venue, framing, and extractor slant. P2 and P3 must keep these channels distinct.

### 3.5 Fairness is not primarily token parity

Primary fairness is based on:

- same source content;
- same temporal cutoffs;
- same splits;
- same extraction substrate;
- matched parameter budget;
- matched compute/FLOPs;
- matched schedules where applicable.

Token parity applies only within text-rendered arms and text-projection comparisons.

### 3.6 Geometry is experimental and ablatable

Learned geometry is in-plan, but it is not assumed true. Geometry must be:

- gauge-invariant at the semantic/output boundary;
- ablatable;
- compatible with geometry-off baselines;
- compatible with context-shuffle controls;
- not allowed to store raw connection matrices as canonical semantic output.

### 3.7 Provider outputs are data construction only

Local or remote LLM providers may be used for substrate construction only. They must never be used in training or evaluation scoring. Provider output must be cacheable, replayable, hashed, provenance-tracked, and fed consistently across arms.

Provider identity and disagreement are lateral context, not invisible preprocessing noise.

---

## 4. Legacy Steps vs v1 Programs

Earlier repository work used “Steps” to build infrastructure. The v1 roadmap uses “Programs” P1–P6.

| Legacy Item | Meaning | v1 Meaning |
|---|---|---|
| Step 1 | repository/schema/storage foundation | prerequisite foundation |
| Step 2 | substrate builder + AXF groundwork | prerequisite foundation, extended by P1/P2 |
| Step 3 | initial training bridge | minimal bridge, not the v1 model |
| Step 4 | falsification artifact preparation | preparation layer, not final evaluation |
| Step 5 | originally first pilot experiment | not complete unless real experiment artifacts exist |
| P1–P6 | current implementation programs | actual path to Axiom v1 |

Do not confuse “Step 3 done” with “the model is done.” Do not confuse “Step 4 done” with “the verdict is done.”

---

## 5. The Six Implementation Programs

```text
P1  Claim-Field Corpus & Provider Ingress
P2  AXF v1 Contract, AXT & AXC-out Specification
P3  AXT Compiler & Runtime Data Interface
P4  Structured-Native Model Stack
P5  Learned Geometry & Claim-Field Graph Dynamics
P6  Training, Experiments, Verdict & Operator Runtime
```

P1–P3 build the structured data and interface foundation. P4–P6 build the model, geometry, training, experiment, and research runtime.

---

# P1 — Claim-Field Corpus & Provider Ingress

**Status:** ✅ implementation-complete / accepted as P2 substrate handoff
**Depends on:** repository foundation and AXF groundwork
**Feeds:** P2, P6

## Objective

Build the provider-capable claim-field corpus layer that turns local and provider-assisted source material into replayable, provenance-rich, claim-centric AXF/AXP artifacts.

## Scope

P1 owns:

- provider ingress protocol;
- deterministic provider;
- OpenAI-compatible provider interface;
- Python-callable provider interface;
- dry-run/cache-safe execution;
- cache and replay;
- local-first/provider cascade;
- provider merge/disagreement traces;
- source registry;
- corpus builder;
- synthetic views;
- epistemic proxy construction;
- negative pools;
- gold/evaluation-reference hooks;
- AXP report enrichment;
- ML/software benchmark fixture corpus;
- optional/plugin-gated PDF reader path;
- CLI commands for corpus/provider operations.

## Required P1-to-P2 Handoff

P1 must expose or document these contracts for P2:

```text
provider_trace_schema
cache_replay_schema
cascade_trace_schema
merge_disagreement_trace_schema
source_registry_schema
claim_candidate_schema
claim_family_schema
relation_candidate_schema
synthetic_view_schema
epistemic_proxy_schema
negative_pool_schema
gold_reference_hook_schema
AXP_enrichment_paths
```

## Non-goals

P1 does not define AXT v1, AXC-out v1, model architecture, training, learned geometry, or final evaluation.

## Acceptance Posture

P1 is accepted as implementation-complete for roadmap purposes. Any remaining corpus-quality improvements are not allowed to block P2 unless they change the provider/claim/relation/negative/gold-reference contracts.

---

# P2 — AXF v1 Contract, AXT & AXC-out Specification

**Status:** ✅ spec/interface-complete
**Depends on:** P1
**Feeds:** P3, P4, P5, P6

## Objective

Lock the structured interface contracts before compilation, modeling, and training begin.

P2 defines what Axiom structure means. It does not compile tensors, train a model, implement geometry, or run experiments.

## Core Deliverables

P2 must define:

- AXF v1 semantic contract;
- AXC v1 record semantics;
- AXP v1 package semantics;
- AXT v1 tensor bundle specification;
- AXC-out v1 structured emission schema;
- field registry;
- vocabulary registry;
- target registry;
- loss-mask contract;
- target-availability contract;
- missing-target semantics;
- relation-neighborhood contract;
- negative-sampling metadata contract;
- provider trace contract;
- provider disagreement as lateral context;
- temporal-mask contract;
- text-projection contract;
- interpreter boundary contract;
- gauge-invariant geometry observable slots.

## P1 Artifacts Formalized by P2

P2 must explicitly formalize how the following P1 artifacts enter AXF/AXT/AXC-out:

```text
provider traces
cache/replay records
cascade traces
merge/disagreement traces
synthetic views
epistemic proxies
negative pools
gold/evaluation-reference hooks
AXP report enrichment
```

## Required AXC-out Distinction

AXC-out must distinguish at least:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

The interpreter must not become an oracle. Raw emissions must remain scored/stored separately.

## Required Masking Semantics

P2 must define masks for:

```text
loss_mask_text_projection
loss_mask_relation
loss_mask_provenance
loss_mask_stability
loss_mask_uncertainty
loss_mask_future_summary
loss_mask_geometry
loss_mask_provider_context
availability_mask_relation
availability_mask_provenance
availability_mask_geometry
availability_mask_future_target
temporal_target_only_mask
```

Missing labels are not negative labels.

## Non-goals

P2 does not build:

- AXT compiler;
- runtime dataset;
- model;
- geometry module;
- training loop;
- evaluation runtime.

P2 is an interface-locking plan.

## Handoff to P3

P2 must produce a clear P3 handoff containing:

```text
AXT tensor group list
field registry path
vocabulary registry path
required fixture bundles
expected compiler inputs
expected compiler outputs
shape conventions
mask conventions
negative sample conventions
AXC-out target conventions
```

---

# P3 — AXT Compiler & Runtime Data Interface

**Status:** ✅ implementation-complete / accepted as P4 handoff
**Depends on:** P2
**Feeds:** P4, P5, P6

## Objective

Build the executable bridge from AXF/AXC/AXP artifacts into AXT tensor bundles and runtime data interfaces.

P3 compiles structure. It does not define new semantics beyond P2 and does not build the model.

## Implemented Deliverables

P3 implemented:

- AXT compiler;
- AXT manifest generation;
- safetensors or equivalent tensor persistence;
- field-registry compiler;
- vocabulary snapshots;
- text-projection token tensors;
- claim/core identity tensors;
- epistemic tensors;
- temporal tensors;
- lateral-context tensors;
- provider-context tensors;
- provenance tensors;
- relation tensors;
- relation-neighborhood tensors;
- negative-sample tensors;
- geometry-slot tensors;
- AXC-out target tensors;
- loss masks;
- availability masks;
- split manifests;
- runtime dataset / batch interface;
- AXT inspection CLI;
- AXT comparison/diff tooling.

## Required Tensor Groups

P3 compiles tensor groups equivalent to:

```text
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

## Critical Separation

P3 preserves:

```text
time != lateral context
provider != source != community != method context
text projection != native substrate
AXC-out target != text target
missing target != negative target
```

## Non-goals

P3 does not build:

- structured-native model;
- learned geometry implementation;
- training loop;
- verdict runtime;
- new semantics outside the P2 contract.

## Handoff to P4/P5/P6

P3 produces a runtime interface that P4/P5/P6 can consume without parsing AXC manually.

The accepted P3 handoff is [P3_AXT_COMPILER_HANDOFF_TO_P4.md](P3_AXT_COMPILER_HANDOFF_TO_P4.md).

The output should make model work look like:

```text
batch = AxtRuntimeDataset(...)[i]
model(batch.structured_inputs) -> axc_out + text_projection
loss = objective(batch.targets, batch.loss_masks, batch.availability_masks)
```

---

# P4 — Structured-Native Model Stack

**Status:** ✅ implementation-complete / accepted as P5/P6 handoff
**Depends on:** P3
**Coordinates with:** P5
**Feeds:** P6

## Objective

Build the primary structured-native LLM model stack.

P4 turns AXT runtime batches into structured latent states, AXC-out emissions, and text projections. It must not collapse back into a normal token-only causal LM.

## Scope

P4 includes:

- structured encoder;
- claim identity encoder;
- epistemic feature encoder;
- provenance encoder;
- relation/hypergraph conditioning;
- relation-neighborhood conditioning;
- provider-context conditioning;
- temporal encoder;
- lateral-context encoder;
- full-complexity core;
- epistemic router;
- structured decoder;
- AXC-out raw emission head;
- AXC-out validation hook;
- text projection head;
- interpreter boundary integration;
- geometry integration hooks;
- geometry-off compatibility;
- text-only baseline compatibility.

## Model Boundary

The model boundary is:

```text
structured AXT batch -> structured latent state -> AXC-out + text projection
```

Not:

```text
text tokens -> causal LM -> text tokens
```

## Required Output Layers

P4 must emit or reserve:

```text
raw_axc_out_emission
structured_relation_logits
provenance_pointer_logits
epistemic_state_predictions
future_summary_structured_target
geometry_observable_predictions
text_projection_logits
```

## Non-goals

P4 does not implement the full learned geometry math if assigned to P5, does not implement the full training runtime, and does not decide the research verdict.

---

# P5 — Learned Geometry & Claim-Field Graph Dynamics

**Status:** ✅ implementation-complete / accepted as P6 handoff
**Depends on:** P3
**Coordinates with:** P4
**Feeds:** P6

## Objective

Build the learnable claim-field geometry and graph dynamics module as a real trained component.

P5 implements the geometry hypothesis while preserving ablatability and gauge-invariant reporting.

## Scope

P5 includes:

- claim-field graph runtime;
- relation/hyperedge graph path;
- context-transition graph;
- learnable connection;
- parallel transport;
- holonomy observables;
- curvature observables;
- spectrum observables;
- geometry-enabled and geometry-disabled paths;
- context-shuffle compatibility;
- parameter-matched non-geometric baseline support;
- torch-native in-loop implementation;
- optional JAX reference/precompute path;
- gauge-invariant output boundary.

## Geometry Contract

P5 may learn internal connection-like objects, but canonical semantic outputs must be gauge-invariant.

Allowed semantic outputs include:

```text
curvature_score
holonomy_norm
trace_summary
spectrum_summary
transport_inconsistency
context_lability
```

Disallowed canonical outputs include:

```text
raw_connection_matrix_as_semantic_field
unablated_geometry_claim
HKR_proof_claim
```

## Non-goals

P5 does not claim HKR is proven. It implements a learnable geometry module and the controls needed to test whether it helps.

---

# P6 — Training, Experiments, Verdict & Operator Runtime

**Status:** next implementation program
**Depends on:** P4 and P5, with P1 gold/evaluation-reference hooks
**Produces:** first Axiom v1 research verdict

## Objective

Build the full execution layer: training, experiment orchestration, controls, scoring, reports, operator interface, and final proceed/redesign/kill verdict.

P6 is where Axiom becomes an experimentally usable structured-native LLM training framework.

## Scope

P6 includes:

- multi-objective training runtime;
- curriculum / complexity ramp;
- checkpointing;
- seed and reproducibility control;
- optimizer and scheduler configuration;
- structured-primary losses;
- secondary text-projection loss;
- loss-mask application;
- target-availability handling;
- negative-sample objectives;
- experiment arm orchestration;
- flat text and structured text baselines;
- structured-native arms;
- no-provenance / no-relations / no-context ablations;
- geometry-off control;
- context-shuffle control;
- popularity/frequency control;
- provider-context ablations;
- scoring on AXC-out;
- scoring on text projection;
- calibration metrics;
- run manifests;
- reproducible research reports;
- operator CLI/API surface;
- final proceed/redesign/kill report.

## Required Experiment Separation

P6 must separate the effects of:

```text
structured input
structured losses
new model architecture
learned geometry
provider-context signals
text projection
```

The verdict must not conflate these effects.

## Core Arms

P6 should include at least:

```text
A flat_text next-token baseline
B structured_text next-token baseline
C structured_text + auxiliary supervision baseline
D structured_native without learned geometry
E structured_native with learned geometry
F structured_native with context shuffle
G structured_native with popularity/frequency controls
H structured_native with provider-context ablation
```

Exact names may differ, but the effects must be separable.

## Verdict Output

P6 must emit a research artifact containing:

```text
run manifests
input hashes
AXT hashes
model configs
training configs
arm definitions
metrics
ablation results
control results
failure cases
uncertainty notes
proceed/redesign/kill recommendation
```

## Non-goals

P6 does not overclaim. It does not state that Axiom proves HKR, proves objectivity, or wins universally. It produces a bounded experimental verdict.

---

## 6. Dependency Graph

```text
P1 -> P2 -> P3 -> P4 -> P6
              \       /
               -> P5 -

P1 also feeds P6 through gold/evaluation-reference hooks.
```

Expanded:

```text
P1 Claim-Field Corpus & Provider Ingress
  -> P2 AXF v1 / AXT / AXC-out Contract
    -> P3 AXT Compiler & Runtime Data Interface
      -> P4 Structured-Native Model Stack
      -> P5 Learned Geometry & Claim-Field Graph Dynamics
        -> P6 Training, Experiments, Verdict & Operator Runtime
```

P4 and P5 may be developed concurrently after P3, but P6 must not begin until their interfaces are stable enough to train and ablate.

---

## 7. Status Table

| Program | Status | Current Action |
|---|---|---|
| **P1 Claim-Field Corpus & Provider Ingress** | ✅ complete / accepted | Substrate foundation for P2/P6 |
| **P2 AXF v1 Contract, AXT & AXC-out Specification** | ✅ spec/interface-complete | Consumed by P3 |
| **P3 AXT Compiler & Runtime Data Interface** | ✅ implementation-complete | Handoff to P4/P5/P6 |
| **P4 Structured-Native Model Stack** | ✅ implementation-complete | Handoff to P5/P6 |
| **P5 Learned Geometry & Claim-Field Graph Dynamics** | ✅ implementation-complete | Handoff to P6 |
| **P6 Training, Experiments, Verdict & Operator Runtime** | ☐ not started | Implement training/experiment runtime next |

---

## 8. First-Class Must-Haves

The following are required for v1 and must not be silently dropped:

1. **Structured-native I/O** with AXC-out.
2. **Text projection** to preserve LLM compatibility.
3. **Learned geometry** as a real trained module.
4. **Gauge-invariant geometry outputs only** at semantic boundaries.
5. **Full epistemic objective set**.
6. **Relation-neighborhood conditioning**.
7. **N-ary/hypergraph path kept open**.
8. **Epistemic router / routing or loss-steering mechanism**.
9. **Negative sampling** for relation/provenance/context/temporal objectives.
10. **Loss masks and target-availability masks**.
11. **Provider context and disagreement as lateral context**.
12. **No truth labels**.
13. **Falsification arms that separate structure, supervision, architecture, geometry, and provider effects**.

---

## 9. How Each Program Becomes an Implementation Plan

Each program plan should include:

1. project context;
2. locked decisions;
3. scope;
4. non-goals;
5. package/file layout;
6. required schemas or APIs;
7. required CLI commands where applicable;
8. fixtures/examples;
9. integrated tests and quality gates;
10. handoff artifacts for the next program.

Do not create standalone “validation steps.” Testing and documentation are part of each implementation program.

---

## 10. Immediate Next Action

The next agent should execute:

```text
PLAN 6 — Training, Experiments, Verdict & Operator Runtime
```

P6 must train and compare the P4/P5-compatible system without external LLM calls,
truth labels, temporal leakage, or overclaiming the verdict.
