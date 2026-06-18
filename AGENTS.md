# AGENTS.md

This file gives instructions to AI coding agents working on Axiom.

Axiom is a structured-native LLM training framework. Do not reduce it to a normal token-in/token-out LLM unless the owner explicitly chooses a rollback path.

---

## 1. Source-of-truth order

When files conflict, follow this order:

1. `docs/concept/CONCEPT.md`
2. `docs/concept/DECISIONS.md`
3. `docs/work/IMPLEMENTATION_ROADMAP.md`
4. `spec/AXF.md`, `spec/AXC_*`, `spec/AXP_*`, `spec/AXT_*`
5. current implementation plans in `docs/work/` or equivalent
6. tests and fixtures
7. older architecture/source-material documents

Older documents may still use legacy terms such as HoloCapsule. The public project identity is Axiom.

---

## 2. Project identity

Use these public names:

```text
Axiom
AXF  = Axiom Exchange Format
AXC  = Axiom Capsule Stream
AXP  = Axiom Package
AXT  = Axiom Tensor Bundle
AXC-out = structured Axiom model emission
Claim-State Capsule
Claim Field
structured-native LLM training framework
```

The internal package may still be named `hcaps`. Treat that as a legacy internal package name unless a plan explicitly asks you to migrate it.

`hcaps` is a legacy internal package name; public project identity is Axiom.

Avoid public-facing terms that sound like sci-fi or imply proof. In particular, do not use "Holo" branding in new public docs except when referencing historical source material.

---

## 3. Core thesis to preserve

Axiom tests whether structured claim-field training improves epistemic competence compared with flat text under matched source content, temporal cutoffs, compute, and parameter budgets.

The v1 target is:

```text
AXF / AXP
  -> AXT structured tensors
  -> structured encoder
  -> relation / hypergraph conditioning
  -> learned geometry module
  -> full-complexity core
  -> epistemic router
  -> structured decoder
  -> AXC-out
  -> interpreter
  -> text projection
```

Text is a projection and comparison interface. It is not the only representation.

---

## 4. Non-negotiable rules

Do not violate these unless the owner explicitly overrides them:

1. No binary truth labels.
2. No temporal leakage.
3. Time is not lateral context.
4. Redundancy is not popularity.
5. Geometry is experimental, ablatable, and gauge-invariant.
6. Raw gauge matrices are not canonical semantic outputs.
7. Structured input, structured supervision, and geometry must remain separately ablatable.
8. External LLMs may be used only for data construction, never in the training or evaluation path.
9. Provider identity/slant must be recorded as context.
10. Interpreter repairs must not hide raw model emissions.
11. Missing labels require loss masks; they are not negative examples.
12. Negative samples are required for relation/provenance/context tasks.
13. All research artifacts must be hashable and reproducible.
14. Do not claim model performance without a proper experiment report.
15. Do not simplify away first-class must-haves.

First-class must-haves:

- learned geometry as a real trained module;
- full epistemic objective set and heads;
- relation-neighborhood conditioning with the n-ary/hypergraph path kept open;
- epistemic router / escalation mechanism.

---

## 5. Implementation style

Axiom agents should build real components with sane defaults. Do not leave placeholder classes that pretend to complete a plan.

Every implementation plan must include, as part of the same work:

- code;
- schema updates where needed;
- CLI wiring where relevant;
- configuration examples;
- documentation updates;
- tests;
- fixture updates;
- manifest/hash support where relevant;
- quality-gate fixes.

Validation and tests are not separate roadmap steps, but they are mandatory done criteria.

Do not start P1 unless the pre-P1 documentation alignment audit exists and names P1 as the next action. After that, create the P1 implementation plan before doing P1 runtime work.

---

## 6. Quality gates

Before a task is considered complete, run and fix:

```bash
ruff check .
ruff format .
mypy src
pytest
```

If a dependency prevents running all checks in the current environment, document exactly what could not run and why. Do not invent successful checks.

---

## 7. Data and format rules

### AXC

Use `.axc` for canonical capsule streams. Do not use `.axc.jsonl` in public examples. AXC may be newline-delimited canonical JSON internally, but AXC is the public format.

### AXP

AXP packages should contain manifests, hashes, splits, sources, capsules, and reports. Missing package parts should fail clearly.

### AXT

AXT is the model-facing tensor bundle. It must include both input tensors and target/output tensors.

### AXC-out

AXC-out is the structured model emission. It must preserve:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

Raw emissions must be stored and scored. The interpreter must be deterministic and ablatable.

---

## 8. Fairness rules

Do not use token parity as the sole fairness rule for structured-native arms.

Primary fairness:

```text
same source content
same temporal cutoffs
same splits
same extraction substrate
matched parameter budget
matched compute budget / FLOPs
matched training schedule where applicable
```

Token parity applies only inside text-rendered arms and text-projection losses.

When designing experiments, separate these factors:

1. structured input;
2. structured auxiliary supervision;
3. architecture change;
4. learned geometry.

---

## 9. Temporal rules

Every predictor-side source timestamp must be at or before the capsule's valid cutoff.

Future-facing fields may exist only as targets and must not be rendered into predictor text or predictor tensors.

Forbidden predictor exposure includes:

```text
future_summary
future_label
target_timestamp
target_only
ground_truth
answer_key
```

Time and lateral context are separate axes. Do not merge them into a single context embedding.

---

## 10. Geometry rules

Geometry may be learned, but it must be reported only through gauge-invariant observables.

Allowed canonical observables:

- curvature score;
- transport inconsistency;
- holonomy norm;
- normalized trace;
- spectrum summary;
- context-lability score.

Not allowed as canonical semantic fields:

- raw connection matrices;
- arbitrary basis-dependent gauge parameters;
- unablated geometry-only claims;
- geometry results without context-shuffle controls.

---

## 11. Provider and data-ingress rules

Axiom may use local or remote OpenAI-compatible providers for data construction. This is allowed only in the substrate/data-ingress path.

Provider-backed extraction must be:

- cached;
- hashable;
- replayable;
- manifest-backed;
- config-driven;
- ablatable;
- marked with provider identity.

Record:

```text
provider_id
provider_family
provider_mode
model_name
confidence
escalation_reason
merge_decision
disagreement_set
```

Do not use external LLM APIs inside model training or model evaluation.

---

## 12. Model implementation rules

The target model is custom structured-native PyTorch.

Do not implement the v1 target as only a stock Hugging Face causal LM. Hugging Face models may be used for text baselines, projection heads, or rollback experiments, but not as a substitute for the structured-native architecture.

Required model surfaces:

- structured encoder;
- relation/neighborhood conditioning;
- learned geometry module;
- full-complexity core;
- epistemic router;
- structured decoder;
- AXC-out output;
- interpreter;
- text-projection head.

JAX may be used for reference/precompute geometry math. In-loop learned geometry should be torch-native unless a plan explicitly changes that.

---

## 13. Loss and target rules

Multi-objective training must use explicit loss masks.

Required target availability concepts:

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

Do not treat missing targets as negative targets.

Do not bake the research thesis into the loss. In particular:

- do not encode redundancy as truth;
- do not encode popularity as correctness;
- do not hard-code a geometry success law and then claim it was discovered;
- do not reward future information leakage.

---

## 14. Negative sampling rules

For relation, provenance, and context objectives, include negative and hard-negative samples.

Required samplers or equivalent logic:

- relation negative sampler;
- provenance negative sampler;
- context negative sampler;
- temporal negative sampler;
- near-but-distinct claim sampler;
- same-topic unrelated sampler.

These must be deterministic and manifestable.

---

## 15. Documentation rules

When you change behavior, update documentation in the same work.

Keep docs concise but complete. Do not claim planned components are implemented unless they are implemented.

Use these phrases consistently:

```text
structured-native LLM training framework
claim-state
claim-field substrate
AXF / AXC / AXP / AXT
AXC-out
text projection
falsification-first
```

Avoid ambiguous phrases:

```text
truth label
validated truth
proven axiom
just metadata
plain token stream as primary representation
```

---

## 16. CLI and artifact rules

CLI commands should fail loudly and clearly.

For every generated artifact, prefer:

- manifest;
- input hash;
- output hash;
- version;
- created_at timestamp;
- code/config reference;
- deterministic seed where applicable.

Do not silently overwrite large or important artifacts unless the command has an explicit `--force` style option.

---

## 17. Commit and PR rules

Keep commits scoped to one implementation plan or coherent sub-task.

Use direct commit messages such as:

```text
feat: add AXT target tensor specification
feat: implement structured encoder runtime
fix: prevent future target exposure in capsule renderer
refactor: separate temporal and lateral context tensors
```

PRs should state:

- which implementation program they advance;
- what artifacts were added;
- which quality gates ran;
- which decisions/ADRs are affected;
- any intentional limitations.

---

## 18. Common mistakes to avoid

### Mistake: turning AXC into decorated text

AXC is structured data, not just a prompt template.

### Mistake: treating special tokens as the native substrate

Special tokens are allowed for structured-text baselines and text projection. They are not the structured-native representation.

### Mistake: forgetting AXC-out

The model must emit structured output, not only text.

### Mistake: using interpreter output as model output

Always store and score raw emissions separately.

### Mistake: confusing Step 3 with P3

Legacy Step 3 is a lightweight training bridge. P3 is the structured-native model program. They are not the same.

### Mistake: confusing Step 4 with P6

Legacy Step 4 generates falsification artifacts. P6 is actual evaluation and verdict. They are not the same.

### Mistake: claiming Step 5 is done without experiments

Step 5/P6 requires actual model runs, metrics, and a decision report.

---

## 19. Agent behavior

Be bold in implementation, strict in claims.

Implement the full planned component. Do not reduce scope out of caution. But never claim a result that has not been measured.

When uncertain, preserve the more structured option and add an ablation path.

When a choice affects the thesis, update the decision record rather than burying the change in code.
