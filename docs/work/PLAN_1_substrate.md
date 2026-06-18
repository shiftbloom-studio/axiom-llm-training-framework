# Plan 1 — Substrate & Corpus (implementation plan + agent prompt)

Part of [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) · Concept: [../concept/CONCEPT.md](../concept/CONCEPT.md) · Phase A · Status: ☐ not started

---

## Project kick-off (read first)

You are implementing one part of **Axiom / HoloCapsule**, a falsification-first research
framework testing whether pretraining on structured **claim-field capsules** beats flat
text on epistemic competence. Steps 1–4 are built and green on Python 3.14 (strict Pydantic
schema, substrate builder with a rule-based *bootstrap* extractor, AXF/AXC/AXP formats, a
torch-free falsification harness). No ML exists yet. The full plan is six implementation
plans; this is **Plan 1**, the data foundation everything else trains on.

Read for context: [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md),
[PROGRESS_REVIEW.md](PROGRESS_REVIEW.md), [../concept/CLAIM_FIELD_SUBSTRATE_BUILDER.md](../concept/CLAIM_FIELD_SUBSTRATE_BUILDER.md),
[../concept/DATA_CONTRACT.md](../concept/DATA_CONTRACT.md), [../project-knowledge/HoloCapsule_Project_Knowledge.md](../project-knowledge/HoloCapsule_Project_Knowledge.md) §6–§10.

## Extraction philosophy (internalize this — it shapes every task)

1. **Claim-states, not truth.** Extraction records *observed* claim-states (claim +
   provenance + time + uncertainty), never truth labels. The forbidden-field guard in
   `src/hcaps/format/validation.py` (`truth`/`is_true`/`correct`) stays. There is no
   truth-oracle to certify — so the exponential cost of validating "true axioms" does not
   apply. Epistemic fields are **proxies/observations with confidence**, not verdicts.
2. **Universal interface, pluggable backends.** Extraction is an *interface*; concrete
   extractors are swappable connectors. You build the interface + harness once; backends
   (LLM-assisted, local-model, human-curated, the existing deterministic bootstrap) are
   selected by config. Nothing locks the framework to one extractor.
3. **Extractor variation is contextual signal, not just noise.** The re-identifiable
   fact-core should stay invariant across extractors/sources; the differing rendering
   (including a model's slant) is *context*, recorded as provenance/community. This feeds
   independent-community redundancy and is a built-in falsification axis. The learned
   geometry (Plans 4–5) discovers what is invariant vs context-dependent — you do not
   pre-certify it here. Capture variation faithfully; do not average it away.

## Your mission (scope)

Turn the bootstrap substrate into a **credible, reproducible claim-field substrate** plus
the **first curated corpus**, so that extraction noise does not dominate the downstream
experiment (a documented kill criterion). Domain for the first corpus: **ML / software
benchmark claims** (well-documented, easy to date and source).

Carry the first-class must-have: **synthetic views**.

## What to build

1. **Universal extraction interface.** Finalize/extend the protocols so all extractors emit
   the validated capsule contract: `ClaimExtractor` (exists, `src/hcaps/extraction/claims.py`),
   plus `RelationExtractor`, `EpistemicEstimator`, and `ViewGenerator`. Add a config-driven
   **registry** to select backend(s). Keep the existing `DeterministicClaimExtractor` as the
   offline fallback.
2. **Credible default backend(s) behind the interface.**
   - Default (recommended): **LLM-assisted** extraction of claims, typed/directional
     relations, epistemic proxies, and synthetic views — with **schema-constrained output**
     and full provenance (which model/version/prompt produced each field).
   - First-class reproducible alternative: a **local-model** backend (no external API).
   - Record the extractor identity per capsule so cross-extractor variation becomes context.
   - *Governance:* using an external LLM supersedes the Step-1 "no external LLM API" rule —
     ratify with a new ADR before bulk external use (offer: ADR-0010). The interface +
     local/deterministic backends need no ratification.
3. **Reproducibility for non-deterministic backends.** Cache and **content-hash** every
   model prompt+output; record model id/version/prompt-hash + seed in the build manifest
   (extend `src/hcaps/substrate/manifest.py`). A build must **replay from cache** byte-stable
   without re-calling the model.
4. **Real epistemic proxies (replace hardcoded constants).** In `src/hcaps/substrate/builder.py`,
   replace `ontic_compatibility=0.5`, `transformation_pressure=0.0`, `stability=UNKNOWN`,
   and doc-count "independence" with estimates from the backend — or explicit
   `unknown` with provenance. Redundancy = **independent-community effective count**, never
   popularity/citation/mention count.
5. **Semantic claim-family clustering** (replace greedy lexical in builder + `canonicalize.py`).
   Use embeddings + threshold/ANN for same-family / near-duplicate grouping; keep
   deterministic tie-breaking and hashing so families are reproducible.
6. **Synthetic views generator** (first-class). Per claim: short summary, FAQ, teaching note,
   counterargument, historical/temporal update, and the `future_summary` target. Each marked
   **synthetic, versioned, provenance-linked** to source; future-only fields stay out of
   predictor-visible inputs (temporal guard). These feed baseline B and the capsule arms.
7. **PDF ingestion + broader readers.** Add a Python-3.14-compatible PDF reader to
   `src/hcaps/ingest/readers.py` (currently skips PDF) with page numbers + section recovery;
   preserve raw-byte and normalized-text SHA-256; metadata via sidecars or extraction.
8. **First curated corpus (ML/software benchmark claims).** Source acquisition + licensing +
   temporal cutoffs; 50–500 claim families. Plus a small **human-verified gold subset**
   (~50–100 capsules) reserved for Plan 6 evaluation. Capture cross-extractor variation as
   context/provenance.
9. **Tests (green on 3.14).** Schema-validity of all emitted capsules; forbidden-truth-field
   guard; temporal-cutoff guard; cached-replay determinism; synthetic-view provenance +
   future-field exclusion; family-clustering sanity; PDF reader; redundancy-not-popularity.

## Guardrails (non-negotiable)

- Claim-states only; never emit truth labels. Keep the forbidden-field guard.
- Redundancy ≠ popularity. Temporal cutoffs enforced. Synthetic content always marked +
  traceable. Every signal carries provenance + confidence.
- Everything ablatable; capture extractor variation rather than collapsing it.
- Real components, sane defaults — **no reductions, no fine-tuning** of thresholds/prompts
  beyond what makes a credible first corpus.
- Project guidelines/ADRs are owner-overrulable; if a task would cross one, ask before
  proceeding (notably the external-LLM ratification above).

## Acceptance criteria

`axiom build-substrate` produces, for the ML/software corpus: a validated capsule set +
manifest with credible extraction (selected backend), real epistemic proxies, semantic
claim-families, synthetic views, and PDF support; the run **replays from cache** byte-stable;
a human-verified gold subset exists for Plan 6; all tests green under `pytest`, `ruff`,
`mypy` on Python 3.14.

## References

WP3 + [../project-knowledge/HoloCapsule_Project_Knowledge.md](../project-knowledge/HoloCapsule_Project_Knowledge.md) §6 (relations), §7 (epistemic), §10 (data/synthetic views) ·
[../concept/CLAIM_FIELD_SUBSTRATE_BUILDER.md](../concept/CLAIM_FIELD_SUBSTRATE_BUILDER.md) · [../concept/DATA_CONTRACT.md](../concept/DATA_CONTRACT.md) ·
code: `src/hcaps/{ingest,extraction,substrate}`, contract in `src/hcaps/schema/capsule.py`, guard in `src/hcaps/format/validation.py`.
