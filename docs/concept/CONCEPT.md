# Axiom — Current Concept, Decisions & Guardrails

This document holds the **synthesized current concept** plus the **owner's decisions and
direction**, captured with their meaning preserved. It is the authority on *intent and
direction*. The formal decision record is [DECISIONS.md](DECISIONS.md). Where older docs
(e.g. [ARCHITECTURE.md](ARCHITECTURE.md)'s "Step 1 boundary") conflict with this file or
DECISIONS.md, this file and DECISIONS.md take precedence.

## 1. Mission & posture

- Build a testable research framework for one question: under equal tokens/compute/params,
  does **claim-field structured pretraining** beat flat text on **epistemic competence**?
- **Build the full theoretical model — properly but small.** No reductions; not
  over-fine-tuned. Do not water down the speculative core out of caution. The falsification
  controls exist to *measure the bold idea honestly*, never as a reason to pre-reduce it.
- Test the real thing; never run the verdict on a stripped stand-in.

## 2. Philosophical–physical concept (the HKR core)

- **Claim-states, not truth.** Records observed claim-states (claim + provenance + time +
  uncertainty); never `truth`/`is_true`/`correct`. No validated "axioms" are required, which
  avoids the exponential cost of certifying truth at scale.
- **Fact-core vs lateral context.** The re-identifiable fact-core stays invariant across
  *lateral* context — extractor/model slant, source, community, framing. That variation is
  recorded as provenance/community (gauge/fibers) and is itself a falsification axis.
  Different extractors (e.g. different LLMs) are different context renderings of one core.
- **Time is not lateral context.** Time is the axis along which the fact-core (reality)
  *itself drifts*. What stays fixed are only the outermost invariants we currently hold
  non-shiftable (relativity-level laws), and they *bound* the drift. So time = the base
  evolution axis of the fact-core; lateral context = the fibers around it. Keep these
  distinct; do not flatten time into context. *(Owner correction.)*
- **Geometry/HKR.** Model context-transport via a learnable connection → holonomy /
  curvature / spectrum, reported only as **gauge-invariant observables**. The learned
  geometry *discovers* what is invariant vs context-dependent; nothing is pre-certified.

## 3. Architectural concept (current)

- **Pipeline:** source → substrate builder (claim-field capsules) → tensorizer (AXT) →
  model + conditioning (arms A–D + auxiliary heads) → learned geometry → training
  (multi-objective) → evaluation + falsification → decision.
- **Frameworks:** PyTorch primary (model/training/arms); in-loop **learned geometry is
  torch-native** (PyTorch Geometric); **JAX = reference/precompute only**; a JAX↔torch
  bridge only if profiling forces it. ([DECISIONS.md](DECISIONS.md) ADR-0007/0009;
  diagram [torch_jax_interface.svg](torch_jax_interface.svg))
- **Geometry mode:** fixed/precomputed (Mode A) vs learned-while-training (Mode B). Chosen:
  **torch-native learned (Mode B)** as the thesis-carrying form; geometry is in-plan,
  concurrent, ablatable, gauge-invariant. (ADR-0005/0008/0009)
- **Extraction:** a **universal extraction interface** with pluggable, swappable backends
  emitting credible-enough (not truth-certified) claim-states with provenance + confidence;
  cross-extractor variation recorded as context. Default backend pending ratification
  (LLM-assisted, cached+hashed); reproducible local backend first-class; small
  human-verified gold set for evaluation. (see [../work/PLAN_1_substrate.md](../work/PLAN_1_substrate.md))
- **First corpus domain:** ML / software benchmark claims.

## 4. Owner decisions (meaning preserved, in order)

1. Build the *proper full solution* of the speculative concept; don't bloat; test the full
   theoretical model, not a reduced one.
2. Commit PyTorch **and** JAX in the docs; make the geometry/HKR + claim-field graph
   workstream **in-plan, not optional**, and develop it **concurrently**. (→ ADR-0007/0008)
3. Use **torch-native** geometry; JAX is reference/precompute; bridge only if profiling
   forces it. (→ ADR-0009)
4. The docs' guidelines/ADRs were Codex-generated and are **owner-overrulable**; if a step
   would need to cross one, **ask first**.
5. Distinguish **time** (the fact-core's drift axis, bounded by fixed outer invariants) from
   **lateral context** (gauge); do not flatten time into context.
6. **Six implementation plans** portray the full framework; **requirements are CLOSED**
   after the four first-class must-haves.
7. First corpus domain = **ML / software benchmark claims**.
8. Extraction is a **framework interface + pluggable backends** (claim-states, not truth);
   the default backend decision is pending.
9. **Documentation structure:** `project-knowledge/`, `work/`, `concept/` folders + a
   navigation file in `docs/`; only `README.md` lives outside `docs/`.

## 5. Guardrails (hold unless the owner overrules)

- **Four first-class must-haves:** learned geometry as a trained module; the full epistemic
  objective set + heads; relation-neighborhood conditioning with the n-ary path kept open;
  the epistemic "router" escalation kept (not dropped).
- **Falsification-first:** equal tokens/compute/params across arms; redundancy ≠ popularity;
  only gauge-invariant observables; negative results preserved; geometry always ablatable
  with a parameter-matched non-geometric baseline and a context-shuffle control.
- **No truth labels, ever.**
- **Soft-guidelines policy:** ask the owner before crossing any stated guideline or ADR.

---

Source: synthesizes the owner's direction across the project conversation. Live status and
next step: [../work/IMPLEMENTATION_ROADMAP.md](../work/IMPLEMENTATION_ROADMAP.md).
