# Axiom v1 — Implementation Roadmap

Last updated: 2026-06-18  
Status: consolidated roadmap after the structured-native architecture decision  
Scope: full first version, no reductions, no separate validation/test phases

---

## 0. Purpose

This roadmap defines the implementation sequence for **Axiom v1** after the structured-native architecture decision.

It supersedes the older six-plan roadmap by making the architectural shift explicit:

```text
Axiom is no longer only an LLM pretraining framework with a richer data format.

Axiom v1 is a structured-native epistemic model framework:
AXF/AXP/AXT input -> structured-native model -> AXC-out structured emission -> text projection.
```

The roadmap is intentionally limited to **ten implementation programs**. Each program is an all-or-nothing work package that later becomes a dedicated implementation plan for an execution agent.

There are **no separate validation, test, or review steps** in this roadmap. Correctness checks, smoke runs, conformance checks, and quality gates are implementation obligations inside each program, not standalone roadmap stages.

---

## 1. Governing idea

Axiom v1 keeps full claim-field complexity from input to output.

The primary model boundary is not plain text. The model ingests structured claim-field tensors, computes over a structured latent epistemic state, and emits a structured Axiom output.

Plain text remains important, but only as:

```text
1. a baseline arm;
2. a controlled projection;
3. a human-readable interpretation layer;
4. a comparability bridge to conventional LLM metrics.
```

The v1 target system is therefore:

```text
source corpus
  -> claim-field substrate
  -> AXF / AXP packages
  -> AXT tensor bundles
  -> structured-native encoder
  -> full-complexity core
  -> learned geometry module
  -> epistemic router
  -> structured decoder
  -> AXC-out emission
  -> interpreter + text projection
  -> experiment arm orchestrator
  -> decision artifacts
```

---

## 2. Locked principles

These principles govern all implementation plans.

### 2.1 Structured-native I/O

The primary input and output of the v1 model are Axiom structures, not token streams.

Input:

```text
claim identity
surface/text projections
relation neighborhood
provenance tensors
context tensors
temporal tensors
epistemic state tensors
optional gauge-invariant geometry tensors
```

Output:

```text
AXC-out structured emission
relation predictions
provenance predictions
stability / uncertainty / O-E-T-R predictions
future-state predictions
geometry observables
text projection
```

Text is a secondary projection, not the architectural boundary.

### 2.2 Claim-state, not truth

Axiom stores and predicts claim states under time, context, provenance, uncertainty, contradiction, support, redundancy, and revision.

Axiom does not encode binary truth fields such as:

```text
truth
is_true
correct
ground_truth
proven_true
```

### 2.3 Time is not lateral context

Time is the evolution axis of the claim/fact-core.

Lateral context is the field/community/source/method/provider framing around that core.

The tensor interface must keep temporal features separate from lateral context features.

### 2.4 Geometry is learned, ablatable, and gauge-invariant

The geometry branch is in-plan and trained end-to-end, but it remains experimental.

Canonical outputs may contain only gauge-invariant observables, such as:

```text
curvature score
transport inconsistency
loop norm
Wilson-loop trace summary
spectrum summary
context-lability score
```

Raw connection matrices are not stable semantic AXF/AXC-out fields.

### 2.5 Fairness is re-grounded

The structured-native architecture invalidates token parity as the primary fairness principle.

Primary fairness:

```text
same source content
same temporal cutoffs
same train/holdout splits
same substrate construction policy
matched parameter budget
matched compute / FLOP budget
matched training schedule where applicable
```

Secondary fairness:

```text
token-budget parity applies only within text-rendered arms and the text-projection loss.
```

### 2.6 Separate the causal factors

The experiment design must keep these factors separable:

```text
1. structured input;
2. structured output;
3. auxiliary objectives;
4. learned geometry;
5. relation-neighborhood conditioning;
6. text projection;
7. provider/extractor context;
8. popularity/frequency effects.
```

No implementation plan may silently merge these into one unablated effect.

---

## 3. Roadmap overview

| Program | Name | Primary output |
|---:|---|---|
| **1** | Claim-Field Corpus & Provider Ingress | Curated AXP corpus with provider-aware extraction |
| **2** | AXF v1 Contract, AXT & AXC-out Specification | Stable structured input/output contract |
| **3** | AXT Compiler & Runtime Data Interface | Training-ready tensor bundles and loaders |
| **4** | Structured Encoder & Relation/Hypergraph Conditioning | Native input representation modules |
| **5** | Learned Geometry Core | Torch-native context-transport module |
| **6** | Full-Complexity Core & Epistemic Router | Structured latent reasoning core |
| **7** | Structured Decoder, AXC-out Interpreter & Text Projection | Native output emission stack |
| **8** | Multi-Objective Training Runtime & Curriculum | Trainable v1 system with structured-primary objectives |
| **9** | Experiment Arm Orchestrator & Decision Runtime | Comparable arms, controls, metrics, and decision artifacts |
| **10** | Open Research Runtime & Operator Surface | Usable CLI/config/release surface for real runs |

---

# Program 1 — Claim-Field Corpus & Provider Ingress

## Objective

Build the first serious claim-field corpus construction layer for Axiom v1.

This program upgrades the current deterministic substrate into a provider-aware corpus builder capable of producing a curated **ML/software benchmark claim corpus**.

## Scope

Implement the complete data-ingress and substrate construction pipeline:

```text
source acquisition
PDF / markdown / text / JSONL ingestion
sidecar metadata
provider-backed extraction
local-first cascade
claim extraction
relation extraction
epistemic proxy extraction
synthetic views
provider-context recording
claim-family merging
provenance tracking
AXP package emission
```

## Required capabilities

- Universal OpenAI-compatible provider interface:
  - `base_url`
  - `model`
  - API key handling through config/environment
  - local and remote providers as first-class interchangeable providers
- Generic Python-callable provider interface.
- Deterministic extractor remains available as offline fallback.
- Local-first cascade:
  - local bulk pass;
  - escalation gate for low-confidence / high-impact cases;
  - remote escalation;
  - deterministic merge;
  - ablatable provider gate.
- Provider context recording:
  - provider ID;
  - model name;
  - provider family;
  - local/remote/deterministic/human mode;
  - extraction confidence;
  - escalation reason;
  - merge decision;
  - disagreement set.
- PDF ingestion suitable for the first corpus domain.
- First curated corpus:
  - 50–500 claim families;
  - ML/software benchmark claims;
  - stable temporal cutoffs;
  - provenance references;
  - relation candidates;
  - synthetic views:
    - FAQ;
    - teaching note;
    - counterargument;
    - historical update;
    - compact structured summary.

## Output artifacts

```text
corpora/ml_benchmark_claims_v1.axp/
configs/corpus/ml_benchmark_claims.yaml
src/hcaps/providers/
src/hcaps/substrate/provider_cascade.py
src/hcaps/substrate/merge.py
src/hcaps/ingest/pdf.py
```

## Non-negotiables

- Provider usage is only for data construction.
- Provider outputs are pinned, cached, hashed, and replayable.
- The same produced substrate feeds all experiment arms.
- Provider identity is treated as lateral context, not hidden metadata.
- No binary truth labels.

---

# Program 2 — AXF v1 Contract, AXT & AXC-out Specification

## Objective

Define the complete structured input and output contract before implementing the v1 model.

This is the most important design program. It prevents the model implementation from drifting into an underspecified structured decoder.

## Scope

Upgrade the format/spec layer from AXF v0.1 to the v1 training interface:

```text
AXF package contract
AXC capsule contract
AXT input tensor bundle contract
AXT target tensor bundle contract
AXC-out structured emission contract
loss-mask contract
target-availability contract
negative-sampling contract
interpreter boundary contract
```

## Required capabilities

### AXT input tensor spec

Define tensors for:

```text
token projections
claim identity features
claim-family IDs
claim-state IDs
relation edges
relation types
relation neighborhoods
n-ary relation placeholders
provenance source IDs
provenance span pointers
epistemic scalar features
context features
temporal features
geometry features
provider-context features
split metadata
manifest hashes
```

### Temporal vs lateral context separation

Define separate channels for:

```text
temporal_features:
  valid_as_of
  observed_at
  source_publication_date
  time_delta_to_cutoff
  temporal_holdout_bin
  future_target_mask

lateral_context_features:
  field
  subfield
  source community
  method context
  provider context
  language/register
  venue/source cluster

geometry_context_features:
  context node IDs
  context transition edges
  sampled loops
  transport paths
```

### AXC-out schema

Define a structured output record with at least:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

AXC-out must support:

```text
claim-state reconstruction
future claim-state prediction
relation prediction
provenance recovery
stability prediction
uncertainty prediction
O/E/T/R proxy prediction
geometry-observable prediction
future-summary prediction
text projection
```

### Target availability and loss masks

Define explicit masks:

```text
loss_mask_text_projection
loss_mask_relation
loss_mask_provenance
loss_mask_stability
loss_mask_uncertainty
loss_mask_epistemic_scalars
loss_mask_future_summary
loss_mask_geometry
loss_mask_context
loss_mask_provider_context
```

Missing labels must not silently become negative labels.

### Negative sampling spec

Define samplers for:

```text
relation hard negatives
near-but-distinct claims
same-topic unrelated claims
provenance negative sources
context negative samples
temporal negative samples
provider-context ablations
```

### Interpreter boundary

Define the strict distinction between:

```text
raw model emission
schema-validated AXC-out
interpreter-rendered object
text projection
```

The interpreter must be deterministic, versioned, ablatable, and forbidden from adding evidence or future information not emitted or referenced by the model.

## Output artifacts

```text
spec/AXT_TENSOR_BUNDLE.md
spec/AXC_OUT.md
spec/LOSS_MASKS.md
spec/NEGATIVE_SAMPLING.md
spec/INTERPRETER_BOUNDARY.md
src/hcaps/format/axc_out.py
src/hcaps/tensor/contracts.py
src/hcaps/tensor/masks.py
```

## Non-negotiables

- P3 must depend on this contract.
- AXC-out is not a text serialization.
- Text is a projection head target.
- Geometry fields are gauge-invariant only.
- All structured targets have availability masks.

---

# Program 3 — AXT Compiler & Runtime Data Interface

## Objective

Compile AXP/AXC packages into training-ready **AXT tensor bundles** consumed by the structured-native model.

This program turns the specification from Program 2 into executable data infrastructure.

## Scope

Implement the full compiler and runtime data layer:

```text
AXP -> AXT compiler
AXC -> tensor compiler
BPE/WordPiece tokenizer integration
special-token projection mode
relation-neighborhood sampler
negative samplers
loss masks
target tensors
side-channel tensors
manifest hashes
runtime dataset objects
collators / batch objects
```

## Required capabilities

- Real subword tokenizer behind the existing tokenizer protocol.
- Shared vocabulary and special-token ID registry for text arms and projection heads.
- Special tokens only as controlled projection/serialization, not native substrate.
- AXT bundle writing:
  - `.axt/` directory packages;
  - `.axt.safetensors` where appropriate;
  - manifest JSON;
  - schema snapshot;
  - source hashes;
  - split metadata.
- Training batch objects for:
  - flat text;
  - structured text;
  - capsule text;
  - structured-native input;
  - geometry-on/off;
  - side-channel-on/off;
  - relation-neighborhood-on/off.
- Relation neighborhood sampler:
  - fixed fanout;
  - typed edge filtering;
  - temporal cutoff awareness;
  - n-ary path preserved in interfaces.
- Negative samplers from Program 2.
- Target and loss masks from Program 2.

## Output artifacts

```text
src/hcaps/tensorizer/
src/hcaps/tensor/
src/hcaps/data/runtime.py
src/hcaps/data/batches.py
src/hcaps/data/neighborhoods.py
src/hcaps/data/negative_sampling.py
configs/tensorizer/axiom_v1.yaml
```

## Non-negotiables

- The structured-native model consumes AXT, not raw JSON dictionaries.
- The same source content can compile into every experiment arm.
- Token parity is maintained only inside text/projection arms.
- Target availability masks are mandatory.
- Future-target leakage must be structurally impossible at tensor level.

---

# Program 4 — Structured Encoder & Relation/Hypergraph Conditioning

## Objective

Implement the structured-native input stack that turns AXT batches into latent model states.

This is the structured encode-in side of the spool-up architecture.

## Scope

Build the native encoder modules:

```text
claim identity encoder
claim text projection encoder
epistemic scalar encoder
context encoder
temporal encoder
provider-context encoder
provenance encoder
relation-edge encoder
relation-neighborhood encoder
hypergraph-ready incidence interface
structured batch embedding composer
```

## Required capabilities

- Separate encoders for:
  - temporal features;
  - lateral context features;
  - provenance features;
  - epistemic state features;
  - relation neighborhoods;
  - text projection features.
- Relation-neighborhood conditioning:
  - edge type embeddings;
  - source/target claim-family references;
  - relation confidence embeddings;
  - neighborhood aggregation;
  - n-ary extension points.
- Hypergraph-ready interface:
  - incidence tensor placeholder;
  - hyperedge type embeddings;
  - typed n-ary relation path preserved.
- Structured latent state object shared by downstream modules.
- Flat-text and structured-text baseline encoders remain available for comparison.

## Output artifacts

```text
src/hcaps/model/encoders/
src/hcaps/model/conditioning/
src/hcaps/model/structured_state.py
src/hcaps/model/baseline_encoders.py
```

## Non-negotiables

- Do not collapse all structure into one text string.
- Do not merge time and context.
- Relation-neighborhood conditioning is first-class.
- The n-ary relation path remains open.
- Encoders must support geometry-on and geometry-off variants.

---

# Program 5 — Learned Geometry Core

## Objective

Implement the torch-native learned geometry branch that models context transport and emits gauge-invariant observables.

This program carries the HKR-inspired geometry hypothesis as a real trained module.

## Scope

Build the geometry subsystem:

```text
claim-field graph representation
context transition representation
learnable connection module
parallel transport module
loop sampler
holonomy composition
curvature observable computation
spectrum / trace summaries
geometry regularization hooks
geometry-off matching path
JAX reference/precompute path
```

## Required capabilities

- Torch-native in-loop geometry module.
- Learnable connection over context transitions.
- Gauge-invariant observable outputs only.
- Sampled loops for holonomy/curvature estimation.
- Context-shuffle compatibility.
- Geometry-disabled and parameter-matched non-geometric path.
- JAX reference implementation off the training path.
- Raw connection matrices may exist as debug artifacts, but not as semantic outputs.

## Output artifacts

```text
src/hcaps/geometry/
src/hcaps/geometry_torch/
src/hcaps/geometry_jax_reference/
src/hcaps/model/geometry_adapter.py
configs/geometry/axiom_v1.yaml
```

## Non-negotiables

- Geometry is trained, not just precomputed.
- Geometry remains ablatable.
- Reported values are gauge-invariant.
- Context shuffle must be able to destroy the signal if geometry is meaningful.
- Geometry must not be hard-coded as true.

---

# Program 6 — Full-Complexity Core & Epistemic Router

## Objective

Implement the central structured-native reasoning core and the epistemic router.

This is the full-complexity core between structured encode-in and structured decode-out.

## Scope

Build the core model modules:

```text
structured latent state fusion
claim-field core blocks
epistemic routing / gating
geometry-conditioned routing
relation-neighborhood routing
loss-mask-aware routing
uncertainty-aware escalation
structured residual pathways
ablation switches
```

## Required capabilities

- Structured latent state fusion from all encoders.
- Epistemic router using:
  - uncertainty;
  - evidence strength;
  - stability;
  - transformation pressure;
  - relation density;
  - geometry observables;
  - context lability.
- Router escalation path:
  - low-complexity stable claims;
  - contested/high-curvature claims;
  - provenance-sensitive claims;
  - relation-heavy claims;
  - future-state claims.
- Compatibility with:
  - geometry-on/off;
  - relation-neighborhood-on/off;
  - side-channel-on/off;
  - text-only baselines.
- Structured latent state exported to the decoder and text projection head.

## Output artifacts

```text
src/hcaps/model/core/
src/hcaps/model/router.py
src/hcaps/model/ablation_switches.py
configs/model/axiom_structured_native_small.yaml
```

## Non-negotiables

- The router is first-class, not an optional helper.
- Routing must be ablatable.
- The core may not secretly become a stock causal LM.
- Geometry-conditioned routing must be separable from non-geometric routing.
- Loss masks must steer objective availability without turning missing labels into negatives.

---

# Program 7 — Structured Decoder, AXC-out Interpreter & Text Projection

## Objective

Implement the structured decode-out stack.

This program creates the primary model output: **AXC-out**.

## Scope

Build the output heads, structured decoder, interpreter, and text projection:

```text
AXC-out decoder
relation prediction head
provenance recovery head
stability head
uncertainty head
O/E/T/R proxy heads
future-summary head
geometry-observable head
schema-validity layer
raw emission writer
AXC-out validator
interpreter
text projection head
```

## Required capabilities

- Raw structured emission recording.
- AXC-out schema validation.
- Interpreter that renders structured emissions without adding hidden knowledge.
- Text projection head for comparability with conventional text baselines.
- Separate losses/outputs for:
  - relation prediction;
  - provenance recovery;
  - stability prediction;
  - uncertainty calibration;
  - epistemic scalar prediction;
  - future-state/future-summary prediction;
  - geometry observable prediction;
  - next-token/text projection.
- Output artifact writing:

```text
raw_emission.jsonl
validated_axc_out.axcout
interpreted_projection.jsonl
text_projection.jsonl
```

## Output artifacts

```text
src/hcaps/model/decoder/
src/hcaps/model/heads/
src/hcaps/interpreter/
src/hcaps/format/axc_out_validation.py
configs/model/output_heads.yaml
```

## Non-negotiables

- AXC-out is primary.
- Plain text is secondary.
- Raw model emissions are always stored separately from interpreted projections.
- The interpreter is deterministic, versioned, and ablatable.
- The interpreter cannot add evidence, future information, or inferred correctness not emitted by the model.

---

# Program 8 — Multi-Objective Training Runtime & Curriculum

## Objective

Implement the complete training runtime for the structured-native Axiom v1 model.

This program makes the system trainable under structured-primary objectives.

## Scope

Build the training stack:

```text
training launcher
optimizer configuration
scheduler configuration
checkpointing
seed control
run manifests
compute-budget tracking
multi-objective loss
loss masks
ablation switches
complexity curriculum
text-projection loss
structured losses
geometry losses
logging
restart/resume
```

## Required capabilities

- Structured-primary multi-objective loss:
  - relation loss;
  - provenance loss;
  - stability loss;
  - uncertainty/calibration loss;
  - epistemic scalar loss;
  - future-summary loss;
  - geometry observable loss;
  - text-projection loss as secondary objective.
- Loss availability masks from AXT.
- Per-objective ablation switches.
- Complexity curriculum:
  - start with core claim/surface channels;
  - introduce provenance/context/relation neighborhoods;
  - introduce geometry;
  - introduce AXC-out targets;
  - introduce future-state targets.
- Compute and parameter budget accounting for fair arm comparison.
- Checkpoint format carrying:
  - model state;
  - tokenizer state;
  - AXT manifest references;
  - loss config;
  - geometry config;
  - random seeds.

## Output artifacts

```text
src/hcaps/training/
configs/training/axiom_v1_small.yaml
configs/training/arms.yaml
configs/training/objectives.yaml
configs/training/curriculum.yaml
```

## Non-negotiables

- Next-token/text loss is not the primary objective for the structured-native arm.
- Every objective is ablatable.
- Missing targets are masked, not treated as false.
- Training must support geometry-on and geometry-off variants.
- Compute and parameter budgets are explicit run metadata.

---

# Program 9 — Experiment Arm Orchestrator & Decision Runtime

## Objective

Implement the runtime that executes the full comparable arm matrix and emits decision artifacts.

This is not a separate validation step. It is the system component that operationalizes the falsification-first design.

## Scope

Build the experiment orchestration layer:

```text
arm registry
arm compiler
budget matching
run scheduler
artifact collection
structured metric computation
text-projection metric computation
control generation
decision report generation
proceed / redesign / kill artifact
```

## Required experiment arms

At minimum:

```text
A: flat text, text loss only
B: structured text, text loss only
C: structured text + auxiliary heads
D: structured-native without geometry
E: structured-native with geometry
F: structured-native with geometry + context shuffle
G: structured-native with provenance ablation
H: structured-native with popularity/frequency control
```

Optional extended arms:

```text
provider_context_off
provider_context_shuffle
relation_neighborhood_off
side_channels_off
text_projection_only
geometry_precomputed
geometry_learned
```

## Required outputs

```text
artifacts/experiments/<run_id>/
  run_manifest.json
  arm_manifests/
  checkpoints/
  axc_out/
  interpreted_outputs/
  text_projections/
  metrics/
  decision_report.md
```

## Required metrics

Structured AXC-out metrics:

```text
relation prediction
support / contradiction / supersession classification
provenance recovery
evidence span recovery
stability prediction
uncertainty calibration
epistemic scalar error
temporal stability
outdated-belief handling
geometry observable consistency
AXC-out syntactic validity
AXC-out referential integrity
```

Text projection metrics:

```text
perplexity / token loss
text rendering quality diagnostics
baseline comparability outputs
```

Decision artifact:

```text
proceed
redesign
kill
```

## Output artifacts

```text
src/hcaps/experiments/
src/hcaps/evaluation/
src/hcaps/decision/
configs/experiments/axiom_v1_small_matrix.yaml
```

## Non-negotiables

- Effects must be separable:
  - structured input;
  - structured output;
  - auxiliary objectives;
  - geometry;
  - router;
  - provider context;
  - popularity/frequency.
- The interpreter and raw emissions are scored separately.
- Geometry claims require geometry-off and context-shuffle comparisons.
- Popularity/frequency controls remain mandatory.
- Negative results must produce a complete decision artifact.

---

# Program 10 — Open Research Runtime & Operator Surface

## Objective

Make Axiom v1 usable as an open research framework, not just a collection of internal modules.

This program packages the full v1 path into a coherent operator-facing system.

## Scope

Implement the public runtime surface:

```text
CLI command groups
configuration schemas
reference configs
small corpus package
model config templates
training config templates
experiment config templates
artifact readers
report renderers
documentation navigation
release metadata
citation metadata
```

## Required capabilities

CLI surfaces for:

```text
axiom corpus build
axiom tensorize build
axiom model init
axiom train run
axiom experiment run
axiom axc-out inspect
axiom report render
```

Configuration surfaces for:

```text
provider ingress
corpus construction
AXT compilation
model architecture
geometry
training objectives
curriculum
experiment arms
output reporting
```

Research release artifacts:

```text
README.md
LICENSE
CITATION.cff
docs/ navigation
example corpus manifest
example AXT manifest
example run config
operator guide
architecture diagrams
```

## Output artifacts

```text
src/hcaps/cli.py
src/hcaps/config/
docs/OPERATOR_GUIDE.md
docs/RESEARCH_RELEASE.md
configs/reference/
examples/research_release/
```

## Non-negotiables

- A new contributor can run the small v1 path from config.
- Public docs use Axiom / AXF / AXC / AXT / AXC-out terminology.
- Legacy `HoloCapsule` names remain internal compatibility only.
- The operator surface must expose the structured-native path, not only text arms.
- The project remains honest: no claims of superiority without the decision runtime artifacts.

---

## 4. Dependency order

```text
Program 1  -> Program 2 -> Program 3
                         -> Program 4 -> Program 6 -> Program 7 -> Program 8 -> Program 9 -> Program 10
                         -> Program 5 -----------/
```

Readable sequence:

```text
1. Build the serious substrate.
2. Define the structured input/output contract.
3. Compile it into tensors.
4. Encode the structure natively.
5. Build learned geometry.
6. Build the full-complexity core and router.
7. Decode into AXC-out and text projection.
8. Train the system.
9. Run the experiment arm matrix and emit the decision artifacts.
10. Package the open research runtime.
```

Programs 4 and 5 may proceed concurrently after Program 3 defines the relevant AXT interfaces. Program 6 must not finalize until Program 5 exposes the geometry interface. Program 7 must not finalize until Program 2 defines AXC-out.

---

## 5. How to turn each roadmap program into an implementation plan

Each implementation plan should contain only its assigned program and must include:

```text
1. project context;
2. locked decisions;
3. exact scope;
4. files/modules to create or modify;
5. interfaces consumed;
6. interfaces emitted;
7. configuration files;
8. artifact layout;
9. non-negotiables;
10. agent prompt.
```

Each plan should avoid scope reduction. If a choice is forced, prefer the fuller structured-native path and preserve rollback as an explicit fallback, not as the main target.

---

## 6. Legacy step mapping

The previous repository milestones remain valuable but are no longer the v1 implementation roadmap.

| Legacy item | Meaning now |
|---|---|
| Step 1 schema/store | Foundation already consumed by AXF/AXT work |
| Step 2 substrate/AXF | Replaced by Program 1 + Program 2 upgrades |
| Step 3 training bridge | Minimal bridge; superseded by Program 3 AXT runtime |
| Step 4 falsification prep | Preparation layer; superseded by Program 9 decision runtime |
| Step 5 first experiment | Becomes Program 9's full experiment arm matrix |
| Old P1–P6 | Replaced by this ten-program v1 roadmap |

---

## 7. Final target

At the end of Program 10, Axiom v1 should be able to run a small but real structured-native experiment:

```text
curated AXP corpus
  -> AXT tensor bundle
  -> structured-native model
  -> learned geometry branch
  -> epistemic router
  -> AXC-out decoder
  -> text projection
  -> arm matrix
  -> decision artifact
```

The desired result is not a predetermined win.

The desired result is a reproducible answer to whether structured-native claim-field modeling improves epistemic competence beyond flat text, structured text, auxiliary losses, popularity effects, and geometry-free baselines under matched content, compute, and parameters.
