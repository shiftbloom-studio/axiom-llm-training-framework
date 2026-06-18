# HoloCapsule Claim-Field Pretraining — Project Knowledge Base

This document is the consolidated project memory for the HoloCapsule Claim-Field Pretraining programme. It is meant to be copied into the project repository as `docs/PROJECT_KNOWLEDGE.md` and kept under version control.

The goal is not to implement a toy demonstration. The goal is to create a serious experimental framework for testing whether large language model pretraining improves when the basic training unit is not a flat token stream, but a structured claim-field capsule carrying text, provenance, epistemic state, relations, temporal context, and optional geometric/HKR-inspired observables.

## 1. Core thesis

Standard LLM pretraining mostly treats the world as a sequence of tokens. This forces the model to infer, indirectly and inefficiently, which spans are claims, which claims are supported or contradicted, which statements are outdated, which facts are stable, which texts are mere duplication, which sources are independent, and which relationships are causal, evidential, rhetorical, or historical.

The HoloCapsule hypothesis is that pretraining can become more data-efficient and epistemically competent if training examples are represented as structured claim-field units.

A HoloCapsule is not a chunk. It is a re-identifiable knowledge object.

A capsule should preserve the raw textual surface while also exposing:

- normalized claim identity,
- multiple surface forms,
- provenance,
- temporal cutoff,
- relation hyperedges,
- evidential status,
- uncertainty,
- independent redundancy,
- context/fiber information,
- training targets,
- and optional HKR/geometric observables.

In compact form:

```text
Pretraining_Data =
  surface_text
+ claim_identity
+ relation_hypergraph
+ provenance
+ uncertainty
+ temporal_state
+ context_fiber
+ latent_memory_key
+ future_summary_target
```

The central experimental question is:

```text
Given equal model size, equal token budget, equal compute, and controlled data quality,
does a HoloCapsule-based training substrate improve epistemic behavior beyond flat-text and structured-text baselines?
```

## 2. Scientific posture

This programme is falsification-first.

It does not claim to prove HKR as physics, metaphysics, or a theory of nature. It tests whether HKR-inspired operational structure has predictive and training value for real epistemic data.

The strongest credible claim reachable by this project is:

```text
A claim-field representation with provenance, temporal structure, independent redundancy,
relations, uncertainty, and optional geometric context improves prediction, calibration,
contradiction handling, source sensitivity, and knowledge-revision behavior in LLM-style training.
```

A negative result is not failure if it is diagnostic. It may show that:

- the capsule schema is too rigid,
- extraction noise dominates the benefit,
- side-channel conditioning is ineffective,
- flat text already captures most of the signal,
- the HKR/geometric observables are decorative,
- or the evaluation harness is not measuring the relevant capability.

The implementation must make these failure modes distinguishable.

## 3. Why claim-field data may be better than flat text

Flat pretraining requires a model to discover epistemic structure implicitly. This is expensive and often unreliable.

A claim-field substrate can expose structure directly:

| Latent property in flat text | Explicit field in HoloCapsule |
|---|---|
| What is being claimed? | `claim` |
| Where did this come from? | `provenance` |
| Is this supported or contested? | `relations`, `epistemic_state` |
| Is this current or historical? | `temporal_cutoff`, `lineage` |
| Is repetition independent? | `redundancy_effective_n` |
| What context does it belong to? | `context_fiber` |
| What changed later? | `training_targets.future_summary` |
| Is this a stable fact or a revision candidate? | `stability`, `transformation_pressure` |

The expected gains are not necessarily lower perplexity alone. The main expected gains are:

- better contradiction detection,
- less hallucinated certainty,
- better provenance sensitivity,
- better historical revision handling,
- better long-context synthesis,
- improved calibration,
- better scientific reasoning,
- and more controllable data ablations.

## 4. HoloCapsule as canonical data object

A HoloCapsule should be a strict, versioned, validated object.

Minimum conceptual fields:

```text
capsule_id
schema_version
created_at
updated_at
claim
surface_forms
epistemic_state
provenance
relations
context
training_targets
quality
lineage
```

A useful canonical shape:

```json
{
  "capsule_id": "cap_example_001",
  "schema_version": "0.1.0",
  "claim": {
    "claim_id": "claim_example_001",
    "canonical_text": "A normalized claim statement.",
    "claim_type": "scientific_claim",
    "language": "en"
  },
  "surface_forms": {
    "original_spans": [],
    "summary": "...",
    "teaching_note": "...",
    "counterargument": "...",
    "table_form": null
  },
  "epistemic_state": {
    "ontic_compatibility": 0.7,
    "evidential_anchoring": 0.6,
    "transformation_pressure": 0.4,
    "redundancy_effective_n": 3.2,
    "uncertainty": 0.25,
    "stability_label": "contested_rising"
  },
  "provenance": [],
  "relations": [],
  "context": {
    "domains": ["example_domain"],
    "time_window": "...",
    "community_ids": [],
    "embedding_keys": []
  },
  "training_targets": {
    "next_token_text": "...",
    "future_summary": "...",
    "relation_targets": [],
    "provenance_targets": [],
    "stability_target": null
  },
  "quality": {
    "extraction_confidence": 0.8,
    "license_status": "known",
    "deduplication_group": null
  },
  "lineage": {
    "pipeline_version": "...",
    "source_manifest_id": "...",
    "parent_capsule_ids": []
  }
}
```

The exact schema should evolve, but the following boundaries should remain stable:

- raw text is not the same thing as claim identity,
- citation count is not the same thing as independent redundancy,
- popularity is not the same thing as evidential anchoring,
- current truth is not the same thing as historical status,
- relation labels are not optional decoration,
- temporal cutoffs are part of the data contract,
- and geometry/HKR fields are experimental observables, not required truth labels.

## 5. Storage philosophy

The storage layer is not a passive file dump. It is part of the learning substrate.

The project should eventually support a hybrid substrate:

| Storage aspect | Near-term implementation | Long-term direction |
|---|---|---|
| Canonical capsule records | JSONL | Parquet/Arrow, object store |
| Numeric training fields | sidecar arrays | tensor-native layout |
| Relations | typed records | hypergraph incidence store |
| Provenance | manifest + source records | probabilistic provenance store |
| Embeddings | sidecar vector files | Matryoshka/hierarchical latent memory |
| Dataset versions | manifests + hashes | lakehouse-style reproducibility |

The first serious implementation should use JSONL and Parquet because they are inspectable, reproducible, and easy to test. A graph database or vector database should not be introduced before the data contract stabilizes.

## 6. Relation layer

Relations should be typed and ablatable.

Initial relation types:

```text
supports
contradicts
extends
refines
supersedes
is_replicated_by
uses_method_from
shares_evidence_with
same_claim_family_as
near_but_distinct_from
```

Relations are not only for retrieval. They become training signal:

- predict whether two claims support or contradict,
- identify supersession/revision,
- distinguish same claim family from same topic,
- recover evidence paths,
- and condition the collator on local claim neighborhoods.

A later hypergraph representation should support n-ary relation events, such as:

```text
method M applied by group G to dataset D supports claim C at time T under context Θ
```

Binary edges are acceptable at Step 1, but the schema should not prevent n-ary extension.

## 7. Epistemic state layer

The initial epistemic state can use bounded scalar proxies. These are not truth labels.

Core dimensions:

| Field | Meaning |
|---|---|
| `ontic_compatibility` | how compatible the claim is with the currently accepted local ontology |
| `evidential_anchoring` | how strongly the claim is anchored in independent evidence/reuse/replication |
| `transformation_pressure` | how much the claim restructures the surrounding field |
| `redundancy_effective_n` | effective number of independent communities carrying the claim |
| `uncertainty` | estimated epistemic uncertainty |
| `stability_label` | categorical state such as `emerging`, `contested`, `established`, `declining`, `superseded` |

Important: redundancy must not be raw citation count, raw mention count, or raw duplication count. It should approximate independent support across distinct communities, methods, groups, or contexts.

## 8. Temporal discipline

Temporal leakage control is mandatory.

Every capsule used for a prediction task needs a cutoff.

At cutoff time `t`, inputs may include only information available at or before `t`. Future targets may refer to events after `t`, but they must be marked as targets and never included in the input side.

This matters for:

- future summary prediction,
- claim stability prediction,
- citation/review/textbook absorption prediction,
- revision/supersession prediction,
- and any scientific-history evaluation.

The schema and tests should catch obvious leakage at the object level before training begins.

## 9. HKR/geometric layer

The HKR-inspired layer is valuable as an inductive bias, not as a forced conclusion.

Possible fields:

```text
context_transition_id
loop_id
curvature_score
holonomy_trace
holonomy_norm
spectrum_summary
context_lability
```

Rules:

- geometry must be optional,
- geometry must be ablatable,
- raw gauge matrices should not be reported as meaningful outputs,
- reported quantities should be gauge-invariant summaries,
- no redundancy law should be hard-coded into the loss,
- context-label shuffle must be able to destroy a real geometric signal,
- and a parameter-matched non-geometric baseline must be included.

The project should first test whether capsules help. Only then should the gauge/holonomy machinery be added as a serious experimental branch.

## 10. Data generation and curation

The project should avoid arbitrary chunking as the final training unit. Chunking is allowed as an ingestion intermediate only.

Preferred progression:

```text
source document
-> normalized document record
-> spans
-> claim candidates
-> claim families
-> relation candidates
-> HoloCapsules
-> trainable tensor batches
```

Important data-quality mechanisms:

- exact deduplication,
- near-duplicate detection,
- paragraph-level boilerplate removal,
- semantic deduplication,
- cluster-balanced sampling,
- license/provenance recording,
- source manifests,
- and temporal cutoffs.

Synthetic views are allowed and likely important, but they must be marked as synthetic, versioned, and traceable to source material.

Recommended synthetic views per claim:

```text
original_span
canonical_claim
short_summary
teaching_note
expert_note
assumptions_table
counterargument
historical_update
future_summary_target
minimal_formal_schema
```

The synthetic layer should complement real data, not replace it.

## 11. Training interface

The training interface should separate three cases:

```text
flat_text_baseline
structured_text_baseline
holocapsule_side_channel
```

This separation is crucial. If HoloCapsules beat flat text only because the text was rewritten into cleaner tutorials, then the gain belongs to structured text, not to the side-channel claim-field hypothesis.

The collator should eventually emit:

```text
input_ids
attention_mask
labels
loss_masks
capsule_ids
claim_ids
relation_tensors
provenance_tensors
epistemic_scalars
context_embeddings
retrieval_neighbor_ids
objective_masks
```

Initial model coupling should be conservative:

- text-only autoregressive baseline,
- structured-text autoregressive baseline,
- side-channel adapter added to a small transformer,
- auxiliary heads for relation/provenance/stability,
- later optional HKR/gauge routing.

Do not start with MoE or a large architecture. The architecture should be complex only after the data substrate proves signal.

## 12. Training objectives

The project should retain next-token prediction as the base objective and add auxiliary objectives gradually.

Candidate loss:

```text
L =
  L_next_token
+ λ_mtp L_multi_token_prediction
+ λ_fsp L_future_summary_prediction
+ λ_rel L_relation_prediction
+ λ_prov L_provenance_recovery
+ λ_cal L_uncertainty_calibration
+ λ_ctx L_context_consistency
```

Important constraints:

- Do not hard-code the HKR redundancy law into the objective.
- Do not reward raw popularity as truth.
- Do not mix future target fields into input fields.
- Make every auxiliary loss optional and ablatable.

## 13. Evaluation and falsification

The first serious experiment should compare at least:

```text
A: flat text baseline
B: structured text baseline
C: HoloCapsule text-only
D: HoloCapsule side-channel
E: D with shuffled context labels
F: D without provenance
G: D without relations
H: popularity/degree/recency baseline
```

Evaluation dimensions:

- validation loss/perplexity,
- relation prediction,
- support/contradiction classification,
- provenance recovery,
- uncertainty calibration,
- temporal stability prediction,
- outdated-belief handling,
- long-context synthesis,
- and contamination/leakage checks.

Kill or redesign criteria should be explicit:

- If structured text beats capsules and side channels add nothing, prioritize synthetic view generation over capsule architecture.
- If context shuffle does not hurt geometric/HKR variants, the geometry is probably decorative.
- If popularity baselines explain most gains, redundancy/provenance modeling is flawed.
- If extraction confidence dominates all metrics, improve substrate builder before model work.
- If no capsule variant beats flat text under equal tokens, the core hypothesis must be redesigned or narrowed.

## 14. Programming-language decision

Python is the correct primary language.

Reason:

- PyTorch ecosystem,
- data tooling,
- tokenizer tooling,
- evaluation tooling,
- scientific computing,
- rapid ablation iteration,
- and easier Codex collaboration.

Rust/C++/Triton should be introduced only after profiling proves a bottleneck.

Initial Python stack:

```text
pydantic v2
orjson
pyarrow
polars
duckdb
pytest
ruff
mypy
typer
rich
```

Later training stack:

```text
torch
transformers or custom tokenizer bridge
accelerate or lightning/fabric only if useful
datasets only if it does not constrain the data contract
wandb/mlflow/local experiment registry
lm-eval or custom evaluation harness
```

## 15. Repository source-material policy

The original research files should be included in the repository, but not as runtime dependencies.

Recommended location:

```text
docs/source-material/
```

Recommended practice:

- keep original filenames or normalized equivalents,
- add a `README.md` explaining what each source contributed,
- record checksums,
- do not import PDFs or Markdown research notes from application code,
- derive stable project docs from them,
- and keep the derived docs under review.

The source files are evidence and project memory. They should not become an implicit hidden dependency of tests or training.

## 16. First implementation target

The first implementation target is Step 1:

```text
Repository constitution + HoloCapsule contract + storage + validation + docs + tests + CI
```

This creates the stable base for all future work.

Step 1 must not produce a model, training result, or benchmark claim.

A clean first commit message:

```text
feat: establish HoloCapsule data contract and research repo foundation
```
