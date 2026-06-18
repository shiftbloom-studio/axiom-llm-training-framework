# Axiom Concept

**Axiom is a structured-native LLM training framework.**

It preserves language-modeling compatibility through a text-projection head, but its primary training substrate is not a flat token stream. Axiom trains models on claim-field structure: claim identity, temporal scope, provenance, relations, epistemic state, context, and optional gauge-invariant geometry.

Text remains important. It is the projection layer, the comparison interface, and one evaluation surface. It is not the only representation.

---

## 1. Purpose of this document

This document is the conceptual source of truth for Axiom v1. It defines the research thesis, the model category, the core abstractions, the architectural target, and the non-negotiable guardrails.

If older documents describe Axiom as a classical text-token LLM pipeline, this document supersedes that framing. If implementation plans need details not specified here, they should extend this concept without reducing its scope.

Axiom should be built small first, but not conceptually watered down. The first version should test the full idea at minimal scale rather than a convenient stand-in that cannot answer the thesis.

---

## 2. One-sentence thesis

Axiom tests whether a model trained with structured claim-field input, structured internal computation, and structured output can achieve stronger epistemic competence than flat text training under matched source content, temporal cutoffs, compute, and parameter budgets.

---

## 3. What Axiom is

Axiom is a research framework for training language-capable models on structured epistemic data.

It is:

- a structured-native LLM training framework;
- a claim-field pretraining framework;
- a data-format and tensor-interface experiment;
- a neuro-symbolic / neural-structured modeling program;
- a falsification-first research environment;
- a framework for testing whether richer knowledge structure improves model behavior.

Axiom is not:

- a normal Hugging Face CausalLM wrapper;
- a RAG system;
- a benchmark leaderboard project;
- a model that asserts truth labels;
- a proof that structured pretraining is better;
- a metaphysical proof of HKR;
- a framework where popularity or repetition equals correctness.

Axiom remains in the LLM family because it trains language-capable models, keeps a tokenizer and text-projection path, can score text perplexity, and can generate readable language. It differs from a classical LLM because its primary I/O is structured claim-field representation rather than token sequence alone.

Canonical phrasing:

```text
Axiom is a structured-native LLM training framework.
It keeps language-modeling compatibility through a text-projection head, while its primary model I/O is structured epistemic data.
```

---

## 4. Core motivation

Classical pretraining flattens knowledge:

```text
source document -> text tokens -> next-token prediction
```

This discards or hides the structure that makes knowledge usable:

- what the claim is;
- when it is valid or observed;
- what sources support it;
- which claims contradict or supersede it;
- whether support is independent or merely repeated;
- which context or community carries it;
- how stable, uncertain, novel, or transformative it is;
- whether a statement is historical, current, contested, or superseded.

Axiom starts from a different premise:

```text
source material
  -> claim-state substrate
  -> AXF / AXC / AXP records
  -> AXT structured tensors
  -> structured-native model
  -> AXC-out structured emission
  -> text projection / interpreter output
```

The goal is not to decorate text with metadata. The goal is to remove the plain-text bottleneck at both the input and output boundary.

---

## 5. The no-fade principle

Axiom follows the **no-fade principle**:

> Full claim-field complexity should be present before model computation begins, remain available inside the model core, and be emitted structurally at the output boundary.

The model should not fade in from plain text, compute internally, then fade out into plain text again. That would preserve the standard bottleneck. Axiom instead uses a structured encode-in, a full-complexity core, and a structured decode-out.

The intuition can be described as spool-up and spool-down:

```text
structured encode-in  ->  full-complexity core  ->  structured decode-out
```

Plain text is a projection surface. It is useful for comparison, interpretability, and language output, but it should not cap the native model representation.

Important scientific caveat:

```text
Structured input + structured output does not automatically imply better knowledge.
It removes an artificial representation bottleneck. Whether that improves model behavior must be measured by the falsification harness.
```

---

## 6. Conceptual foundations

### 6.1 Claim-states, not truth labels

Axiom records claim-states, not eternal truths.

A claim-state is a temporally scoped, context-sensitive representation of a claim and its epistemic state. It may be emerging, contested, supported, established, superseded, fragmented, retracted, or field-dependent.

Forbidden fields include:

```text
truth
is_true
correct
is_correct
factuality
ground_truth
proven_true
```

Axiom may record evidence, support, uncertainty, contradiction, and stabilization. It must not encode binary truth as a training shortcut.

### 6.2 Fact-core versus lateral context

The re-identifiable claim core is not the same as its surface expression.

The same claim may appear as:

- an abstract sentence;
- a review summary;
- a benchmark table;
- a contradiction;
- a later historical note;
- a superseded formulation;
- a method reuse;
- a source span;
- a structured relation.

Axiom separates claim identity from surface realization.

Lateral context includes:

- field or subfield;
- community;
- source or venue;
- method context;
- language and register;
- extractor/provider slant;
- graph neighborhood;
- relation environment.

Lateral context is the fiber-like context around a fact-core.

### 6.3 Time is not lateral context

Time is not just another context label.

Time is the drift axis of the claim-state itself. It determines whether a source was available, whether a target is future-facing, whether a claim was plausible at a cutoff, and whether later evidence may be used.

Axiom must keep these separate:

```text
temporal axis       = valid_as_of, observed_at, source date, target date, drift
lateral context     = field, community, method, source, provider, relation neighborhood
```

Do not flatten time into context. Doing so destroys the temporal leakage discipline and weakens the HKR-inspired distinction between drift and lateral transport.

### 6.4 Geometry is experimental and gauge-invariant

Axiom reserves a learned geometry branch for context transport, holonomy, curvature, and stability observables.

Geometry is:

- in-plan;
- ablatable;
- learned where possible;
- never assumed true;
- never reported as raw gauge matrices;
- only reported through gauge-invariant observables.

Allowed geometric outputs include:

- curvature score;
- transport inconsistency;
- holonomy norm;
- normalized trace;
- spectrum summary;
- context-lability metrics.

Raw learned connection matrices are implementation artifacts, not stable semantic outputs.

---

## 7. Core data abstractions

### 7.1 AXF

**AXF** is the Axiom Exchange Format family. It is the public data contract for claim-centric pretraining substrates.

AXF is not a single opaque binary format. It is a versioned format family built from inspectable, hashable, reproducible artifacts.

### 7.2 AXC

**AXC** is the Axiom Capsule Stream.

It contains claim-state capsules. A capsule stores the structured state of a claim at a temporal and contextual position.

AXC may be encoded as newline-delimited canonical JSON in v0.x, but `.axc` is the public format suffix. JSONL is the encoding strategy; AXC is the contract.

### 7.3 AXP

**AXP** is the Axiom Package.

It bundles capsules, sources, relations, contexts, splits, manifests, hashes, leakage reports, and control artifacts into a package directory.

### 7.4 AXT

**AXT** is the Axiom Tensor Bundle.

It is the compiled model-facing representation of AXF. AXT is the bridge from symbolic/structured claim-state records into tensors consumed by the structured-native model.

AXT must include both input tensors and target/output tensors.

### 7.5 AXC-out

**AXC-out** is the structured output emitted by the model.

It is not just a text generation result. It is a structured prediction containing model emissions for claim state, relations, provenance, epistemic values, temporal targets, geometry observables, and optional text projection.

AXC-out must support three layers:

```text
raw_emission             = direct model output before repair or interpretation
validated_axc_out        = schema-validated structured output
interpreted_projection   = deterministic interpreter result for display or downstream use
```

Raw emissions must always be preserved for scoring. The interpreter must never hide model errors.

---

## 8. Structured-native model architecture

The v1 target architecture is:

```text
AXF / AXP
  -> AXT compiler
  -> structured encoder
  -> relation / hypergraph neighborhood conditioning
  -> learned geometry module
  -> full-complexity core
  -> epistemic router
  -> structured decoder
  -> AXC-out
  -> interpreter
  -> text projection
```

### 8.1 Structured encoder

The encoder ingests structured tensors, not only token IDs.

Inputs include:

- claim identity representation;
- claim-state features;
- source/provenance features;
- temporal features;
- lateral context features;
- relation neighborhood features;
- side-channel epistemic vectors;
- optional geometry features;
- text tokens where used for projection or baseline comparability.

### 8.2 Relation and hypergraph conditioning

Relations are not decorative metadata. They are model inputs and prediction targets.

The architecture should support:

- binary relations;
- n-ary or hyperedge path kept open;
- local claim-family neighborhoods;
- source spans and evidence pointers;
- relation negative sampling;
- provenance negative sampling;
- context negative sampling.

### 8.3 Learned geometry core

The geometry module learns context transport and emits gauge-invariant observables. It must be parameter-matched and ablatable.

It must be possible to compare:

```text
structured model without geometry
structured model with learned geometry
structured model with geometry but shuffled context
```

### 8.4 Epistemic router

The epistemic router is a routing/conditioning mechanism that uses epistemic state to influence computation.

It may route or modulate based on:

- stability;
- uncertainty;
- contradiction level;
- transformation pressure;
- provenance strength;
- relation density;
- context-lability;
- temporal status.

It should not be removed as a simplification. It is a first-class hypothesis.

### 8.5 Structured decoder

The structured decoder emits AXC-out components. It is the native output surface.

The decoder should predict:

- claim-state status;
- relation candidates;
- support/contradiction/supersession scores;
- provenance references;
- evidence span pointers;
- epistemic proxy values;
- uncertainty/calibration values;
- future-summary or future-state targets when available;
- geometry observables when enabled;
- text-projection hidden state.

### 8.6 Text projection head

The text projection head produces tokens from the same internal state.

It exists to preserve:

- language-modeling compatibility;
- comparison with flat-text baselines;
- perplexity scoring;
- human-readable outputs;
- interoperability with existing LLM evaluation surfaces.

Text projection is secondary. It must not become the primary model objective by accident.

---

## 9. Training objectives

Axiom uses structured-primary multi-objective training.

Possible losses include:

```text
L_total =
  L_structured_claim_state
+ L_relation_prediction
+ L_provenance_recovery
+ L_evidence_span_pointer
+ L_epistemic_value_prediction
+ L_uncertainty_calibration
+ L_temporal_stability_prediction
+ L_future_summary_or_future_state
+ L_geometry_observable_prediction
+ L_text_projection
```

The text projection loss is important but secondary.

### 9.1 Target availability and loss masks

Not every capsule has every target.

AXT must include target-availability and loss-mask tensors such as:

```text
loss_mask_text_projection
loss_mask_relation
loss_mask_provenance
loss_mask_evidence_span
loss_mask_stability
loss_mask_uncertainty
loss_mask_future_summary
loss_mask_geometry
loss_mask_context_transport
```

Missing labels must not silently become negative labels.

### 9.2 Negative sampling

Relation and provenance tasks require hard negatives.

Required samplers:

- relation negative sampler;
- provenance negative sampler;
- context negative sampler;
- temporal negative sampler;
- near-but-distinct claim sampler;
- same-topic unrelated sampler.

Without negative samples, relation heads may learn that everything is related.

### 9.3 No hard-coded victory

The loss must not encode the hypothesis as a guarantee.

For example:

- do not force redundancy to equal truth;
- do not bake a fixed redundancy-curvature law into the loss;
- do not reward popularity as correctness;
- do not give future evidence to the predictor side;
- do not allow the interpreter to repair outputs before scoring raw emissions.

---

## 10. Fairness and experiment arms

The old fairness rule of strict token parity is insufficient for structured-native models.

Primary fairness:

```text
same source content
same temporal cutoffs
same train/validation/test splits
same extraction substrate
matched parameter budget
matched compute budget / FLOPs
matched training schedule where applicable
```

Secondary fairness:

```text
token-budget parity only within text-rendered arms and text-projection losses
```

Axiom must separate three effects:

```text
1. structured input
2. additional structured supervision
3. new architecture / learned geometry
```

The experiment arm matrix should preserve this separation.

Suggested v1 arms:

| Arm | Input | Losses | Geometry | Purpose |
|---|---|---|---|---|
| A | Flat text | Text projection / next-token | None | Classical baseline anchor |
| B | Structured text | Text projection / next-token | None | Tests serialization and special-token effects |
| C | Structured text | Text + auxiliary heads | None | Tests richer supervision without native structure |
| D | Structured native | Same auxiliary heads | Off | Tests native structure without geometry |
| E | Structured native | Same auxiliary heads | On | Tests geometry contribution |
| F | Structured native | Same auxiliary heads | On + context shuffle | Tests whether geometry uses real context |
| G | Structured native | Same auxiliary heads | On + popularity control | Tests redundancy/popularity confound |
| H | Structured native | Reduced channels | Off/on ablatable | Tests channel necessity |

Special tokens belong to text arms and projections. They are not the native Axiom substrate.

---

## 11. Interpreter guardrails

The interpreter is useful but dangerous.

It must be deterministic, versioned, auditable, and ablatable.

Rules:

- raw model emissions must always be stored;
- raw emissions must be scored separately from interpreted outputs;
- interpreter repairs must be logged;
- the interpreter must not add evidence not emitted or referenced by the model;
- the interpreter must not add future information;
- the interpreter must not silently convert malformed output into credit;
- validation failures must remain visible.

Output layers:

```text
raw_emission             scored directly
validated_axc_out        schema and integrity checked
interpreted_projection   display/downstream form
text_projection          language surface
```

---

## 12. Data construction and providers

Axiom may use model-backed data construction, including local and remote LLM-compatible providers, but only in the data-ingress path.

Allowed:

- local-first provider cascade;
- remote escalation for low-confidence or high-impact items;
- deterministic extractor fallback;
- human-curated gold sets;
- cached, hashed, replayable extraction results.

Not allowed:

- external LLMs in training loops;
- external LLMs as evaluation judges for headline metrics unless explicitly isolated and marked;
- hidden provider behavior that changes the substrate without manifests;
- untracked provider slant.

Provider identity is itself a lateral context signal and must be recorded:

```text
provider_id
provider_family
provider_mode
model_name
base_url category
confidence
escalation_reason
merge_decision
disagreement_set
```

Provider effects should be ablatable.

---

## 13. First corpus domain

The first corpus domain is ML/software benchmark claims.

This domain is useful because it contains:

- claims about model performance;
- benchmark results;
- method comparisons;
- supersession and revision;
- replication-like signals;
- dataset and evaluation dependencies;
- strong temporal leakage risks;
- high popularity/confound pressure.

This makes it a good early stress test for claim-state modeling.

---

## 14. Implementation programs

Axiom v1 is implemented through all-or-nothing programs. Validation, tests, docs, and quality gates are part of each program's done criteria, not separate roadmap steps.

Canonical v1 programs:

1. Claim-Field Corpus & Provider Ingress
2. AXF v1 Contract, AXT & AXC-out Specification
3. AXT Compiler & Runtime Data Interface
4. Structured Encoder & Relation/Hypergraph Conditioning
5. Learned Geometry Module
6. Full-Complexity Core & Epistemic Router
7. Structured Decoder, AXC-out Interpreter & Text Projection
8. Multi-Objective Training Runtime & Curriculum
9. Experiment Arm Orchestrator & Decision Runtime
10. Open Research Runtime & Operator Surface

The next implementation program is P1: Claim-Field Corpus & Provider Ingress. The most important pre-model contract is the AXF/AXT/AXC-out interface; the model must not be built against vague structured targets.

---

## 14.1 Legacy steps versus v1 programs

Older documents use Step 1-5 language. That vocabulary is historical and must not override the v1 program map.

```text
Legacy Steps 1-4 = foundation and infrastructure
Legacy Step 3    = lightweight training bridge, not P3
Legacy Step 4    = falsification artifact preparation, not P9/P10 verdict
Legacy Step 5    = not complete without model runs, metrics, and decision report
```

P1 is the next implementation program after the pre-P1 documentation alignment pass. Archived Plan 1/Plan 2 documents are not the active P1/P2 implementation plans.

---

## 15. Required specification before model build

Before the structured-native model is implemented, Axiom must specify:

- AXT input tensor schema;
- AXT target tensor schema;
- AXC-out schema;
- loss-mask schema;
- relation neighborhood sampling;
- negative sampling;
- temporal-feature separation;
- lateral-context feature separation;
- geometry-observable schema;
- interpreter boundaries;
- raw-emission scoring;
- fairness arm matrix.

The model should not be allowed to define these implicitly.

---

## 16. Success and failure

Axiom succeeds if it produces a reproducible answer to the thesis, not if the answer is positive.

Possible outcomes:

- structured-native wins and survives controls;
- structured text explains most gains;
- auxiliary losses explain most gains;
- geometry adds no measurable value;
- context shuffle does not hurt geometry;
- popularity/frequency controls explain the signal;
- the structured model is too expensive or unstable;
- the hypothesis must be narrowed or rejected.

Negative results are useful results. The framework should make them visible.

---

## 17. Non-negotiable guardrails

1. Axiom is structured-native, not text-native.
2. Text is a projection and comparison interface, not the only representation.
3. No truth labels.
4. Time is not lateral context.
5. Temporal leakage must be blocked and audited.
6. Redundancy is not popularity.
7. Geometry is optional, ablatable, and gauge-invariant.
8. Raw connection matrices are not canonical reported fields.
9. Interpreter output must not hide raw model emissions.
10. Structured input, structured supervision, and geometry must be separably ablated.
11. Missing labels require masks; they are not negatives.
12. Provider slant is recorded as context, not ignored.
13. External LLMs may construct data, not train or evaluate the model path.
14. All artifacts are versioned, hashed, and replayable.
15. Do not reduce the concept to a standard LLM pipeline unless explicitly choosing a rollback path.

---

## 18. Canonical short description

```text
Axiom is a structured-native LLM training framework for claim-field pretraining.
It keeps language-modeling compatibility through a text-projection head, but its primary model I/O is structured epistemic data: claim identity, provenance, relations, temporal scope, context, uncertainty, and optional gauge-invariant geometry.
```
