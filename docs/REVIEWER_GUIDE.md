# Axiom Reviewer Guide (Post-P6)

**Axiom is a structured-native LLM training framework.** It tests whether claim-field structure (identity, provenance, relations, epistemic state, temporal scope, lateral context, provider traces, negative pools, and nullable/maskable gauge-invariant learned geometry) improves epistemic competence compared with flat text under matched source content, temporal cutoffs, splits, extraction substrate, parameter budgets, compute, and schedules.

Text projection and token baselines are mandatory for LLM compatibility and comparability, but they are secondary. The primary model I/O is structured (AXT tensors → structured encoder/core/decoder → raw AXC-out emission → validated/interpreted/text projections).

This repository contains the **complete v1 implementation** (P1–P6) plus post-P6 hardening. It is a real, runnable, small-scale research runtime. It is **not** a scale benchmark or a production system.

## What Was Built (P1–P6)

- **P1**: Provider-aware claim-field corpus ingress (deterministic + OpenAI-compatible providers, cache/replay, disagreement traces, source registry, negative pools, gold-reference hooks, ML/software benchmark fixture corpus, CLI).
- **P2**: AXF v1 / AXC v1 / AXP v1 / AXT v1 / AXC-out v1 contracts, field & vocabulary registries, loss/availability masks, negative sampling, temporal & provider context rules, interpreter boundaries.
- **P3**: AXT compiler (AXP/AXC → structured tensor bundles + safetensors + manifests), runtime dataset/batch interface, inspection & validation CLI, all required tensor groups (claim, epistemic, temporal, lateral, provider, provenance, relations + neighborhoods, negatives, geometry slots, text projection, targets, masks).
- **P4**: Torch-native structured model stack (field embeddings, structured encoder, relation/provenance/provider/context conditioning, full-complexity core, epistemic router, structured decoder, raw AXC-out emission head, text projection head, geometry hooks, ablations, serialization).
- **P5**: Learned claim-field geometry module (graph batches, lateral transitions, learnable connections, parallel transport/holonomy, gauge-invariant observables only — curvature, holonomy norm, transport inconsistency, etc. — geometry-off and context-shuffle controls, P4 integration).
- **P6**: Multi-objective training runtime, experiment orchestrator (suite of arms + controls), scoring on raw AXC-out + text projection, verdict generation, operator (inspect/export/compare runs), smoke and mini configs, reproducibility manifests (including AXT input snapshotting).

All programs include their tests, CLI surfaces, configs, and handoff artifacts. Post-P6 hardening tightened language (geometry is native + ablatable, not "optional"), added input-artifact snapshotting and persisted comparison outputs, and aligned docs.

## Reproduce the Local Research Runtime (Smoke / Mini)

All commands use the uv setup described in the root README (or `./bin/axiom`).

Typical end-to-end for a smoke experiment (no GPU required):

```bash
# 1. Compile a tiny AXT bundle (P3)
uv run axiom axt compile \
  --input examples/axf/v0_1/minimal_dataset.axp \
  --output artifacts/axt/minimal-smoke.axt \
  --config configs/axt/compile_smoke.yaml \
  --allow-all-without-split --force

# 2. Smoke a training arm (P4/P5/P6)
uv run axiom train run \
  --axt artifacts/axt/minimal-smoke.axt \
  --config configs/training/smoke_structured_native_geometry.yaml \
  --output-dir runs/smoke-struct-geom \
  --seed 42

# 3. Run a small experiment suite (orchestrates multiple arms + ablations)
uv run axiom experiment run \
  --config configs/experiments/smoke_suite.yaml \
  --output-dir runs/smoke-suite \
  --seed 13

# 4. Score + verdict + inspect
uv run axiom score run runs/smoke-suite/<run_id>
uv run axiom verdict generate runs/smoke-suite/<run_id> --thresholds configs/verdict/default_claims_ladder.yaml
uv run axiom run inspect runs/smoke-suite/<run_id>
uv run axiom experiment compare runs/smoke-suite/<run_id>
```

See `configs/experiments/mini_axiom_suite.yaml` and `configs/training/mini_structured_native.yaml` for the next size up (still local).

All runs produce manifests, input hashes, AXT snapshots (post-hardening), metrics, and operator artifacts.

## Key Guardrails (Non-Negotiable)

- No binary truth labels (claim-state only; gold references are evaluation anchors only).
- Time ≠ lateral context (temporal scope is the drift axis; provider/source/community/method/framing are separate lateral fibers).
- Geometry outputs must be gauge-invariant observables at the semantic boundary. Raw connection matrices are never treated as canonical semantic fields.
- Structured input, auxiliary structured supervision, architecture, and learned geometry are separately ablatable.
- Provider/LLM calls are data-construction only (P1). Never used in training, scoring, or verdict.
- Raw AXC-out emissions are always stored and scored separately from any interpreter or text projection.
- Missing targets use loss masks; they are not negative examples.
- Negative samples are required for relation/provenance/context objectives.
- Fairness = matched content + cutoffs + splits + substrate + param/compute budget (token parity only inside text-rendered arms).
- Reproducibility via manifests, content hashes, seeds, and (post-P6) copied AXT input artifacts per run.

See `AGENTS.md`, `docs/concept/CONCEPT.md`, `docs/concept/DECISIONS.md`, and the specs in `spec/` for the full set.

## What This Is Not (Yet)

- No large-scale training runs or benchmark numbers.
- No claim that the hypothesis is proven.
- No external LLM usage inside the training/evaluation loop.
- Smoke and mini configs are runtime/integration checks and control proofs, not performance evidence.
- The internal package is still `hcaps` (legacy); public identity is Axiom / AXF / AXC / AXP / AXT / AXC-out.

## Where to Look in the Code

- `src/hcaps/axt/` — compiler, dataset, batch collator, registries, inspection.
- `src/hcaps/model/` — structured encoder, core, router, decoder, AXC-out head, text projection, parameter counting.
- `src/hcaps/geometry/` — graph dynamics, transport, observables, ablation modes, reference parity.
- `src/hcaps/training/` — AxiomTrainer, multi-objective loss with masks, checkpointing.
- `src/hcaps/experiments/` — suite config, orchestrator, arms, controls, comparison.
- `src/hcaps/scoring/`, `verdict/`, `operator/` — AXC-out + text scoring, verdict ladder, run export/inspect.
- `configs/` — smoke, mini, geometry-on/off, ablation suites (all the controls live here).
- `tests/` — especially `tests/axt/`, `tests/model/`, `tests/geometry/`, `tests/training/`, `tests/experiments/`.

Quality gates (must stay green):

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

(195 tests as of post-P6; 1 non-blocking PyTorch nested-tensor prototype warning.)

## Source of Truth Order (for any proposed change)

1. `docs/concept/CONCEPT.md`
2. `docs/concept/DECISIONS.md`
3. `docs/work/IMPLEMENTATION_ROADMAP.md`
4. `spec/*` (AXF_V1, AXT_V1, FIELD_REGISTRY_V1, etc.)
5. Current handoff docs in `docs/work/`
6. Tests + fixtures
7. Older material (never overrides active docs)

## Current Next Action (per roadmap)

Review `docs/work/POST_P6_HARDENING_AUDIT.md`. Practical follow-ups include preparing external-review / GPU-sponsor materials (this guide + one-pager), hardening operator UX, adding more existing-behavior tests, or owner-selected scale-up experiments. Do not start new P7-style scope without an updated roadmap entry and ADR if it touches first-class must-haves.

Questions about the thesis, an ablation, or a specific arm? The code + configs + run manifests are the primary artifacts. Start by running a smoke suite and inspecting the produced `run_manifest.json`, scored outputs, and verdict report.

This guide is intentionally short. The real depth is in the running system and the specs.
