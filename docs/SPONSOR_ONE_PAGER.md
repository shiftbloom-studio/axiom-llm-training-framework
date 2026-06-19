# Axiom — Structured-Native LLM Training Framework

**One-page sponsor / external reviewer brief** (post-P6, June 2026)

**Visual one-pager (recommended for sharing)**: `docs/assets/axiom-sponsor-onepager.jpg` (generated for high visual appeal and legibility as a single-page handout or slide).

The Markdown below is the canonical text source — easy to convert to PDF (`pandoc docs/SPONSOR_ONE_PAGER.md -o axiom-onepager.pdf`) or hand to a designer. Keep the two in sync if you edit.

Axiom tests a precise hypothesis: **does claim-field structure improve epistemic competence vs. flat text**, under strict fairness (identical source content, temporal cutoffs, splits, extraction substrate, parameter budgets, compute/FLOPs, and schedules)?

It is **not** another wrapper around a causal LM. The primary substrate and model I/O are structured.

## The Core Thesis (Preserved)

AXF/AXP corpus → AXT structured tensors → structured encoder + relation / provenance / provider / context conditioning → native learned geometry (ablatable) → full-complexity core + epistemic router → structured decoder → raw AXC-out emission → validated / interpreted / text projections.

Text projection is **mandatory** for LLM compatibility and head-to-head comparison, but it is secondary. All first-class must-haves (learned geometry as real module, full epistemic objectives, relation-neighborhood conditioning, n-ary path open, epistemic router, loss masks, negative sampling, no truth labels, provider = data construction only) are implemented and ablatable.

## What Exists Today (Complete v1 Runtime)

- Full P1–P6 implementation + post-P6 hardening pass.
- Reproducible local smoke and mini experiment suites (P6) with separable arms:
  - Flat text baseline
  - Structured text baseline
  - Structured + auxiliary supervision
  - Structured-native (geometry off)
  - Structured-native (learned geometry on)
  - + context shuffle, provider ablation, popularity controls, etc.
- End-to-end: corpus ingress (with provider traces) → AXT compiler → model training (masked multi-objective losses) → AXC-out scoring + text projection metrics → verdict ladder → operator inspect/export/compare.
- Everything is manifest-backed, hashable, seed-controlled, and (post-P6) snapshots key AXT inputs per run directory.
- ~195 tests, ruff + mypy + pytest green. Python 3.14 only.

**No performance claims are made.** Smoke/mini are runtime and control verification, not benchmarks.

## How to Run It Yourself (Zero Mystery)

```bash
uv python install 3.14
uv sync --python 3.14
uv sync --python 3.14 --reinstall-package axiom-llm-training-framework

# Compile a tiny bundle and run a smoke training arm
uv run axiom axt compile --input examples/axf/v0_1/minimal_dataset.axp \
  --output artifacts/axt/minimal.axt --config configs/axt/compile_smoke.yaml --allow-all-without-split --force

uv run axiom train run --axt artifacts/axt/minimal.axt \
  --config configs/training/smoke_structured_native_geometry.yaml \
  --output-dir runs/my-smoke --seed 42

# Or run a full controlled experiment suite
uv run axiom experiment run --config configs/experiments/smoke_suite.yaml \
  --output-dir runs/smoke-expt --seed 13
```

See `docs/REVIEWER_GUIDE.md` and `docs/work/POST_P6_HARDENING_AUDIT.md` for details.

## Key Differentiators

- Geometry is **native, nullable, maskable, ablatable** — not a decorative add-on. Only gauge-invariant observables (curvature, holonomy, transport inconsistency, spectrum, context lability) are semantic outputs.
- Strict separation of time (drift) vs. lateral context (provider, source, framing...).
- Raw model emissions (AXC-out) are always scored separately from interpreter or text projection.
- Negative sampling and explicit loss/availability masks are first-class.
- Provider slant is treated as observable lateral context, never invisible preprocessing.
- Falsification-first design: controls are built to separate the effects of structure, supervision, architecture, and geometry.

## Limitations (Stated Clearly)

- Small scale only (local CPU/GPU smoke & mini configs).
- No external LLM calls inside training or evaluation.
- No "truth labels" — only claim-state prediction and evaluation references.
- The research verdict on the hypothesis is still to be run at interesting scale.

## Artifacts & Reproducibility

Every run produces:
- `run_manifest.json` + copied AXT input artifacts
- Per-arm metrics (structured AXC-out + text projection)
- Scoring, verdict, and comparison JSONs
- Full config + seed + code references

All research artifacts are designed to be hashable and replayable.

## The Ask

We are looking for:
- GPU / compute sponsorship to run the mini and medium suites at proper scale with matched budgets.
- External review of the implementation against the stated guardrails and thesis.
- Feedback on the operator / experiment UX before larger runs.

**Repository**: https://github.com/shiftbloom-studio/axiom-llm-training-framework  
**Start here**: `README.md`, `docs/REVIEWER_GUIDE.md`, `docs/work/IMPLEMENTATION_ROADMAP.md`, `docs/concept/CONCEPT.md`

Run the smoke suite locally in < 5 minutes on a laptop. The code either satisfies the structured-native contract or it doesn't — the artifacts make it auditable.

*Be bold in implementation. Strict in claims.*
