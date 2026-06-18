# Post-P6 Project Hardening, Self-Review & Quick-Win Pass

## Role

You are GPT-5.5 XHigh / Codex acting as the **senior implementation reviewer, research-consistency auditor, and final project hardening agent** for the Axiom repository.

You are not here to redesign the project.
You are not here to reduce scope.
You are not here to replace the structured-native approach with a conventional LLM pipeline.
You are here to make the implemented system **coherent, robust, internally consistent, testable, documented, and ready for serious research use** after the six implementation programs have landed.

Repository:

```text
shiftbloom-studio/axiom-llm-training-framework
```

Expected branch name:

```text
chore/post-p6-hardening-review
```

Expected commit message:

```text
chore: harden Axiom after P6 implementation
```

Expected PR title:

```text
Post-P6 hardening, self-review, and consistency cleanup
```

---

## Project identity to preserve

Axiom is a **structured-native LLM training framework**.

It remains LLM-compatible through:

- text projection,
- token baselines,
- text-rendered arms,
- secondary text losses,
- readable interpreted outputs,
- and comparability to conventional language-model baselines.

But its primary model substrate is **not** a flat token stream.

Its primary substrate is structured claim-field representation:

- claim identity,
- claim-state,
- temporal scope,
- lateral context,
- provenance,
- relation neighborhoods,
- epistemic state,
- provider/extractor context,
- negative pools,
- target availability masks,
- loss masks,
- learned claim-field geometry,
- and structured AXC-out emission.

Text is a projection and comparison interface. It is not the system boundary.

The core target architecture is:

```text
AXF / AXC / AXP corpus
  -> AXT structured tensor bundle
  -> structured-native encoder
  -> relation / provenance / provider / context conditioning
  -> native learned geometry
  -> full-complexity core + epistemic router
  -> structured decoder
  -> raw AXC-out emission
  -> validated AXC-out
  -> interpreted projection
  -> secondary text projection
```

---

## Non-negotiable innovation drivers

Do not weaken or bypass the following. These are not optional details; they are the reason the project exists.

### 1. Structured-native model I/O

The v1 target model is structured-native. It must not collapse into:

```text
AXC -> capsule text -> tokenizer -> ordinary CausalLM
```

Capsule text, special tokens, and text renderings are allowed as:

- comparison arms,
- debugging surfaces,
- text-projection inputs/outputs,
- fallback baselines,
- or ablations.

They are not the native Axiom substrate.

### 2. Geometry is native, not decorative

Geometry is first-class and native in Axiom v1.

Use this exact mental model:

```text
Geometry is native, not decorative.
Ablatable, not optional.
```

Correct language:

```text
Geometry is native to the v1 target model, nullable at the data-record level,
maskable at the tensor/loss level, and ablatable at the experiment level.
Only gauge-invariant observables are exported as semantic outputs.
```

Incorrect language:

```text
optional geometry
geometry add-on
geometry metadata only
future optional HKR module
```

Those phrases are allowed only if explicitly referring to:

- record-level availability,
- missing geometry masks,
- geometry-off control arms,
- or experimental ablation.

### 3. AXF / AXT / AXC-out are primary interfaces

Preserve the distinction:

```text
AXF = format family
AXC = claim-state capsule stream
AXP = package
AXT = compiled tensor bundle / runtime model interface
AXC-out = structured model emission
```

Do not blur AXC-out into ordinary JSON logging or text output.
Do not treat AXT as just tokenized text plus metadata.

### 4. Text projection is mandatory but secondary

Axiom remains an LLM training framework because text projection remains mandatory.

Do not remove:

- tokenizer path,
- text-projection head,
- secondary next-token/text loss,
- text-rendered baseline arms,
- text quality/perplexity comparability.

But do not elevate text projection back to the primary model objective.

### 5. No truth labels

Axiom stores claim-states, not truth.

Forbidden as semantic fields:

```text
truth
is_true
correct
is_correct
factuality
proven_true
label_truth
```

If a file needs evaluation anchors, prefer:

```text
evaluation_reference
gold_reference
human_verified_reference
reference_span
expected_relation_type
```

These must not mean metaphysical truth.
They are evaluation references only.

### 6. Time and lateral context are distinct

Do not flatten time into context.

Time is the drift/evolution axis of claim-state.
Lateral context is the fiber/context dimension around the claim-state:

- provider/extractor,
- community,
- source,
- method,
- benchmark family,
- venue,
- language/register,
- field/subfield,
- framing.

AXT, model conditioning, geometry, and evaluation must preserve this distinction.

### 7. Provider ingress is data construction only

Local or remote providers may be used for data construction/substrate harvesting only.

They are forbidden in:

- model training,
- evaluation scoring,
- benchmark judging,
- verdict generation,
- hidden postprocessing of model outputs.

Provider outputs must be:

- cached,
- hashed,
- replayable,
- provenance-tracked,
- versioned,
- and fed equally into all relevant arms.

Provider identity/slant is lateral context and should remain ablatable.

### 8. Interpreter must not become an oracle

The AXC-out interpreter must not secretly repair model outputs using hidden knowledge.

Always preserve and score separately:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

Interpreter rules must be deterministic, versioned, auditable, and ablatable.

No interpreter may add evidence, future information, hidden provenance, or relation targets not emitted or explicitly referenced by the model.

### 9. Fairness rule

Primary fairness is not equal token count.

Primary fairness is:

- same source content,
- same temporal cutoffs,
- same splits,
- same extraction substrate,
- matched parameter budget,
- matched compute/FLOPs,
- matched training schedule where applicable.

Token parity applies only within:

- text-rendered arms,
- flat vs structured text comparisons,
- text-projection losses/metrics.

### 10. Effects must remain separable

Do not mix these effects without ablation:

```text
structured input
additional structured objectives
new architecture
learned geometry
provider-derived substrate
text projection
interpreter postprocessing
```

If a quick fix hides which effect caused a result, it is not allowed.

---

## Purpose of this pass

Perform a complete post-P6 repository preparation, self-code review, bug-fixing, cleanup, and quick-win pass.

This pass happens after the implementation roadmap has been executed through P6.

It should make the repository:

- internally coherent,
- runnable,
- reproducible,
- logically aligned with the structured-native concept,
- ready for serious research use,
- and ready for external review or GPU-sponsor inspection.

This is not a new roadmap stage.
This is not a redesign.
This is not permission to simplify away the project’s core.

---

## What you may do

You may:

- fix bugs,
- fix broken imports,
- fix failing tests,
- fix CLI issues,
- fix broken docs links,
- improve type safety,
- improve error messages,
- improve manifests,
- improve config defaults,
- improve reproducibility checks,
- improve deterministic behavior,
- improve small performance bottlenecks,
- add missing tests for existing behavior,
- add small helper utilities,
- remove dead code,
- consolidate duplicate code,
- improve README/AGENTS/docs consistency,
- add missing audit reports,
- add missing sanity checks,
- add missing smoke configs,
- add missing fixture coverage,
- harden provider cache/replay safety,
- harden AXT/AXC-out validation,
- harden geometry-off and context-shuffle controls,
- harden interpreter boundaries,
- harden no-truth-label checks,
- improve CI reliability.

You may also implement **quick wins** if and only if they satisfy all rules below.

---

## Quick-win policy

Quick wins are allowed.

But a quick win must be a **small, local, high-confidence improvement** that strengthens the existing concept. It must not introduce a new research direction or bypass a hard architectural decision.

### Allowed quick wins

Examples of allowed quick wins:

```text
Add missing CLI subcommand aliases for already implemented functionality.
Add a manifest hash check that was clearly intended but missing.
Add a no-truth-label recursive check to a new artifact type.
Add a better error when an AXP package lacks required files.
Add a small test fixture for AXC-out validation.
Add a deterministic seed to an existing sampler.
Add a small doc note clarifying geometry-native vs geometry-ablatable.
Add missing target/loss mask consistency checks.
Add a sanity metric for side-channel missing rate.
Add shape assertions for AXT tensor groups.
Add a dry-run flag to a provider command if the infrastructure already exists.
Add warnings when text projection is present but structured outputs are missing.
Add a small helper to compare two AXT manifests.
Add a small config example for geometry-off ablation.
Add a smoke test for raw_emission -> validated_axc_out -> interpreted_projection.
```

### Forbidden quick wins

The following are not quick wins. They are prohibited unless the owner explicitly approves a new ADR or roadmap change.

```text
Replace structured-native model with a stock Hugging Face CausalLM target.
Treat AXC-out as normal text generation.
Make geometry an optional plugin rather than a native first-class module.
Remove geometry-on as the primary structured-native arm.
Remove geometry-off or parameter-matched non-geometric controls.
Remove text projection.
Make next-token loss the primary objective again.
Remove loss masks or target-availability masks.
Treat missing targets as negative labels.
Use provider/LLM calls during evaluation or scoring.
Add hidden interpreter repairs.
Add truth/is_true/correct labels.
Flatten time into context.
Hard-code HKR laws such as redundancy-curvature scaling into the loss.
Remove popularity/frequency controls.
Remove context/provider shuffle controls.
Collapse AXT into token IDs plus JSON metadata.
Use special tokens as the native representation.
Drop provider traces, cache/replay, or provenance to simplify code.
Rename the entire internal Python package unless specifically assigned.
Add large dependencies without clear need.
Rewrite the architecture around a new framework.
```

When in doubt, prefer no quick win over a risky quick win.

---

## Required review order

Follow this order. Do not start patching randomly.

### Phase 0 — Establish current state

Inspect:

```text
README.md
AGENTS.md
CONCEPT.md
IMPLEMENTATION_ROADMAP.md
DECISIONS.md
spec/**
docs/**
configs/**
examples/**
src/**
tests/**
.github/workflows/**
pyproject.toml
.python-version
LICENSE
CITATION.cff if present
```

Identify:

- current version,
- implemented plans,
- CI status,
- known failing tests,
- known TODOs,
- stale docs,
- duplicate docs,
- stale roadmap references,
- active architecture conflicts,
- missing quick wins.

Do not edit yet except for trivial scratch notes.

### Phase 1 — Conceptual consistency audit

Check active docs and code comments for violations of the non-negotiable innovation drivers.

Search for and review:

```text
optional geometry
standard causal LM
token-in/token-out
plain text boundary
capsule text native
truth
is_true
correct
ground_truth
equal tokens
same token budget
no external LLM calls
HoloCapsule public naming
Step 5 done
P3 done
P4 done
HKR proof
validates HKR
raw connection matrix
```

Fix or annotate active occurrences.

Historical/source-material files may remain contradictory if clearly archived or labeled historical.

### Phase 2 — Architecture integrity review

Review the implemented system along the real dependency chain:

```text
P1 provider/corpus ingress
P2 AXF v1 / AXT / AXC-out contracts
P3 AXT compiler/runtime interface
P4 structured-native model stack
P5 learned geometry and graph dynamics
P6 training/experiments/verdict/operator runtime
```

For each subsystem, check:

- interfaces align with the next subsystem,
- manifests are sufficient,
- hashes are stable,
- identifiers are stable,
- masks exist where targets may be missing,
- configs are explicit,
- CLI is wired,
- examples are current,
- tests cover the public behavior,
- docs match implementation.

### Phase 3 — Bug fixing and hardening

Fix clear bugs and integration failures.

Prioritize in this order:

1. broken imports / package metadata / CLI entrypoints,
2. failing tests caused by implementation mismatch,
3. data-contract violations,
4. AXF / AXT / AXC-out shape/schema mismatches,
5. loss-mask / target-availability errors,
6. temporal leakage risks,
7. provider cache/replay risks,
8. geometry-native / geometry-off ablation risks,
9. interpreter oracle risks,
10. reproducibility and manifest gaps,
11. docs/README/AGENTS mismatch,
12. typing/formatting cleanup,
13. performance quick wins.

Do not spend time polishing style while architectural errors remain.

### Phase 4 — Quick wins

After critical fixes, implement allowed quick wins only if they:

- reduce future agent confusion,
- strengthen reproducibility,
- improve safety against concept collapse,
- improve CLI/operator usability,
- improve test coverage for existing behavior,
- or close obvious small gaps in manifests/audits/configs.

Each quick win must be recorded in the audit report with:

```text
what changed
why it was safe
which guardrail it supports
```

### Phase 5 — Integrated check run

Run appropriate checks.

At minimum, attempt:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

If formatting changes are allowed, run:

```bash
ruff format .
```

If CI uses additional commands, run those too.

If environment limitations prevent checks, record exactly:

- command attempted,
- failure mode,
- whether it appears environment-related or code-related,
- what remains to verify.

Do not hide failing tests.
Do not mark the pass complete if core tests fail due to your changes.

### Phase 6 — Final documentation and audit report

Create:

```text
docs/work/POST_P6_HARDENING_AUDIT.md
```

If that path does not exist, create it.

The report must include:

```text
1. Executive summary
2. Repository state before pass
3. Files inspected
4. Files changed
5. Bugs fixed
6. Quick wins added
7. Conceptual conflicts found and resolved
8. Conceptual conflicts intentionally left because historical/source-material
9. Test and quality-check results
10. Remaining known risks
11. Blockers, if any
12. Confirmation checklist
13. Recommended next action
```

Confirmation checklist must include:

```text
[ ] Axiom remains a structured-native LLM training framework
[ ] Text projection remains mandatory and secondary
[ ] Geometry is native, not decorative; ablatable, not optional
[ ] AXF / AXT / AXC-out remain primary interfaces
[ ] No truth-label contamination
[ ] Time and lateral context remain distinct
[ ] Provider ingress remains data-construction only
[ ] Provider cache/replay/provenance preserved
[ ] Raw AXC-out / validated AXC-out / interpreted projection remain distinct
[ ] No hidden interpreter oracle repairs introduced
[ ] Loss masks and target availability semantics preserved
[ ] Negative sampling semantics preserved
[ ] Fairness rule is content/cutoff/split/params/compute-based
[ ] Token parity limited to text arms/projection
[ ] Geometry-off and non-geometric controls preserved
[ ] Context/provider shuffle controls preserved
[ ] No HKR proof overclaim introduced
[ ] README / CONCEPT / AGENTS / ROADMAP / ADRs aligned
[ ] CI or local quality checks run or limitations documented
```

---

## Files and areas to pay special attention to

### Concept and governance

```text
README.md
AGENTS.md
CONCEPT.md
IMPLEMENTATION_ROADMAP.md
DECISIONS.md
RESEARCH_PROTOCOL.md
EVALUATION_PROTOCOL.md
DATA_CONTRACT.md
FALSIFICATION_HARNESS.md
TRAINING_BRIDGE.md
```

### Format and specs

```text
spec/AXF.md
spec/AXC_CAPSULE_SCHEMA.md
spec/AXP_PACKAGE_LAYOUT.md
spec/AXT_TENSOR_BUNDLE.md
spec/AXC_OUT_SCHEMA.md
```

### P1 provider/corpus

```text
src/hcaps/providers/**
src/hcaps/corpus/**
src/hcaps/ingest/**
src/hcaps/extraction/**
src/hcaps/substrate/**
```

Check especially:

- provider traces,
- cache/replay,
- cascade configs,
- disagreement traces,
- synthetic views,
- epistemic proxies,
- negative pools,
- evaluation-reference hooks.

### P2/P3 data interface

```text
src/hcaps/axt/**
src/hcaps/format/**
src/hcaps/training_bridge/**
```

Check especially:

- field registries,
- vocabularies,
- AXT manifest,
- tensor groups,
- target tensors,
- target availability masks,
- loss masks,
- negative sample tensors,
- text projection tensors,
- AXC-out target tensors.

### P4 model

```text
src/hcaps/model/**
```

Check especially:

- structured encoder,
- relation/provenance/provider conditioning,
- temporal encoder,
- lateral context encoder,
- epistemic router,
- full-complexity core,
- structured decoder,
- AXC-out emission,
- text projection head,
- geometry hook points.

### P5 geometry

```text
src/hcaps/geometry/**
```

Check especially:

- learned connection,
- parallel transport,
- holonomy/curvature observables,
- gauge-invariant exports,
- raw matrix containment,
- geometry-off mode,
- parameter-matched non-geometric controls,
- context/provider shuffle compatibility.

### P6 training/experiments/verdict

```text
src/hcaps/training/**
src/hcaps/experiments/**
src/hcaps/falsification/**
src/hcaps/evaluation/**
src/hcaps/operator/**
```

Check especially:

- structured-primary losses,
- secondary text losses,
- loss masks,
- optimizer/scheduler configs,
- checkpointing,
- experiment arms,
- scoring on AXC-out,
- text-projection metrics,
- verdict reports,
- no provider use in evaluation.

---

## Required final PR body

The PR body must include:

```markdown
## Summary

## What was reviewed

## Bugs fixed

## Quick wins included

## Conceptual guardrails checked

## Tests / checks run

## Remaining risks

## Not done intentionally

## Next recommended action
```

Under “Not done intentionally”, explicitly list any tempting simplifications you refused, for example:

```text
- Did not replace structured-native model with stock CausalLM.
- Did not make geometry an optional plugin.
- Did not collapse AXC-out into text generation.
- Did not use providers in evaluation.
- Did not add truth labels.
```

---

## Definition of Done

This pass is complete only if:

1. the repo is aligned with the structured-native concept;
2. active documentation does not contradict the architecture;
3. obvious bugs and integration issues are fixed;
4. quick wins, if any, are safe and documented;
5. core tests/checks pass or failures are honestly documented;
6. `docs/work/POST_P6_HARDENING_AUDIT.md` exists;
7. the PR clearly states what changed and what remains;
8. no innovation driver was bypassed for convenience.

If you cannot satisfy all of the above, do not claim completion. Mark the PR as partial and list blockers.

---

## Final instruction

Be conservative with claims and ambitious with execution.

Do not reduce the project to something easier.
Do not hide uncertainty.
Do not add new speculative features unless they are tiny, safe quick wins.
Do not let convenience erase the structured-native breakthrough.

The correct final state is not “smaller Axiom”.
The correct final state is **coherent, runnable, research-ready Axiom**.
