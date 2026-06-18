# Post-P6 Hardening Audit

Date: 2026-06-19

## 1. Executive Summary

This pass completed the post-P6 hardening and self-review sweep after P5 and P6 landed on `main`.

The repository remains aligned around Axiom as a structured-native LLM training framework. No implementation-plan scope was reduced, no training/evaluation provider calls were introduced, no truth labels were added, and AXF / AXT / AXC-out remain primary interfaces with text projection preserved as a secondary comparison surface.

The pass fixed active-document wording that still framed geometry as "optional" rather than native, nullable/maskable, and ablatable. It added small reproducibility and operator quick wins: P6 training manifests now snapshot key AXT input artifacts into each run directory, and the standalone experiment comparison CLI now persists pairwise/control comparison JSON files. Focused tests cover both changes.

## 2. Repository State Before Pass

- Branch: `main`.
- Local state before this pass continued from P6 plus an in-progress Post-P6 hardening worktree.
- Recent completed commits before this pass:
  - `8e81b0a feat: add learned claim-field geometry module`
  - `137c844 feat: add training experiments verdict and operator runtime`
- `origin/main` still pointed at the P4 structured-native model stack commit during this local pass.
- P5 and P6 were already committed locally.
- The remaining required work was the Post-P6 hardening audit, safe quick wins, final checks, and a scoped hardening commit.

## 3. Files Inspected

Inspected directly or through recursive searches:

- `README.md`
- `AGENTS.md`
- `.github/workflows/ci.yml`
- `pyproject.toml`
- `.python-version`
- `docs/README.md`
- `docs/concept/CONCEPT.md`
- `docs/concept/DECISIONS.md`
- `docs/concept/DATA_CONTRACT.md`
- `docs/concept/PROJECT_KNOWLEDGE.md`
- `docs/work/IMPLEMENTATION_ROADMAP.md`
- `docs/work/PLAN_4_STRUCTURED_NATIVE_MODEL_STACK.md`
- `docs/work/PLAN_5_LEARNED_GEOMETRY_CLAIM_FIELD_GRAPH_DYNAMICS.md`
- `docs/work/PLAN_6_TRAINING_EXPERIMENTS_VERDICT_OPERATOR_RUNTIME.md`
- `docs/MODEL_STACK.md`
- `docs/GEOMETRY_MODULE.md`
- `docs/TRAINING_RUNTIME.md`
- `docs/EXPERIMENT_ORCHESTRATOR.md`
- `docs/SCORING_AND_VERDICT.md`
- `docs/OPERATOR_RUNTIME.md`
- `docs/P6_HANDOFF_FINAL.md`
- `spec/AXT_TENSOR_BUNDLE.md`
- Active `spec/**` files through targeted conceptual searches
- Active `configs/**` files through targeted search and direct diff review
- `src/hcaps/model/**`
- `src/hcaps/geometry/**`
- `src/hcaps/training/**`
- `src/hcaps/experiments/**`
- `src/hcaps/scoring/**`
- `src/hcaps/verdict/**`
- `src/hcaps/operator/**`
- `tests/model/**`
- `tests/geometry/**`
- `tests/training/**`
- `tests/experiments/**`
- `tests/scoring/**`
- `tests/verdict/**`
- `tests/operator/**`

Searches reviewed active occurrences of stale or risky terms including `optional geometry`, `standard causal LM`, `token-in/token-out`, `truth`, `ground_truth`, `equal tokens`, `no external LLM`, `HoloCapsule`, `HKR proof`, and `raw connection matrix`, excluding clearly historical `docs/project-knowledge/**` and `docs/archive/**` material where appropriate.

## 4. Files Changed

- `README.md`
- `docs/README.md`
- `docs/MODEL_STACK.md`
- `docs/P6_HANDOFF_FINAL.md`
- `docs/concept/DATA_CONTRACT.md`
- `docs/concept/DECISIONS.md`
- `docs/concept/PROJECT_KNOWLEDGE.md`
- `docs/work/IMPLEMENTATION_ROADMAP.md`
- `docs/work/PLAN_4_STRUCTURED_NATIVE_MODEL_STACK.md`
- `spec/AXT_TENSOR_BUNDLE.md`
- `configs/training/smoke_flat_text.yaml`
- `configs/training/smoke_structured_text.yaml`
- `configs/training/smoke_structured_native_no_geometry.yaml`
- `configs/training/smoke_structured_native_geometry.yaml`
- `configs/training/mini_structured_native.yaml`
- `configs/experiments/smoke_suite.yaml`
- `configs/experiments/mini_axiom_suite.yaml`
- `configs/experiments/geometry_controls_suite.yaml`
- `src/hcaps/training/trainer.py`
- `src/hcaps/cli.py`
- `tests/training/test_trainer.py`
- `tests/operator/test_cli_operator.py`
- `docs/work/POST_P6_PROJECT_HARDENING_SELF_REVIEW_PROMPT.md`
- `docs/work/POST_P6_HARDENING_AUDIT.md`

## 5. Bugs Fixed

- Fixed stale active documentation that still described geometry as "optional" in a way that could weaken the Post-P6 guardrail. Active docs now use nullable/maskable language and keep geometry native, gauge-invariant, and ablatable.
- Fixed stale next-action text that still told agents to run the Post-P6 pass after the pass was completed.
- Fixed a reproducibility gap in P6 run manifests: run directories now copy key AXT input artifacts into `run_dir/input/` and reference them from `run_manifest.json`.
- Fixed an operator usability gap in `axiom experiment compare`: standalone compare now persists `comparisons/pairwise_metrics.json` and `comparisons/control_effects.json`, matching the artifact behavior users expect from suite runs.

## 6. Quick Wins Added

- AXT input artifact snapshotting in P6 training.
  - What changed: `AxiomTrainer` copies `axiom.json`, selected `manifests/*.json`, and selected `reports/*.json` from the input AXT bundle into each run directory.
  - Why safe: it only copies existing reproducibility artifacts and records relative paths; it does not alter training data, losses, scoring, or model behavior.
  - Guardrail supported: hashable, replayable, manifest-backed research artifacts.

- Persisted standalone experiment comparison outputs.
  - What changed: `axiom experiment compare` writes pairwise metric deltas and control effects into `run_dir/comparisons/`.
  - Why safe: it serializes deterministic summaries already computed by existing comparison helpers.
  - Guardrail supported: operator usability and reproducible artifact review.

- Explicit geometry provider kind in training and experiment configs.
  - What changed: configs now state `geometry_provider_kind: none` or `geometry_provider_kind: learned`.
  - Why safe: the config value already existed and defaults to `none`; this makes the intended provider path explicit.
  - Guardrail supported: geometry-on, geometry-off, and non-geometric controls remain visible and ablatable.

- Focused tests for quick wins.
  - What changed: trainer tests assert copied input artifacts are recorded and present; operator CLI tests assert comparison JSON files are written.
  - Why safe: tests validate existing intended behavior without adding new runtime semantics.
  - Guardrail supported: future agents cannot silently drop reproducibility or comparison artifacts.

## 7. Conceptual Conflicts Found And Resolved

- Resolved active "optional geometry" wording in README, roadmap/spec/docs, and P4 model docs. Replacement language uses nullable/maskable geometry features or observables.
- Preserved the P5/P6 distinction that geometry is native to v1, ablatable in experiments, and gauge-invariant at semantic/export boundaries.
- Preserved AXC-out raw/validated/interpreted/text distinction in docs and tests.
- Preserved provider policy: provider ingress is data construction only; P6 training, scoring, verdict, and operator paths remain deterministic and local.
- Preserved current internal `hcaps` / `HoloCapsule` compatibility notes while keeping public identity as Axiom.

## 8. Conceptual Conflicts Intentionally Left Because Historical/Source-Material

- Historical source material under `docs/project-knowledge/**` may still use HoloCapsule terminology or older framing. It is explicitly labeled as historical/source material in docs navigation and is superseded by `docs/concept/CONCEPT.md`, `docs/concept/DECISIONS.md`, and the active roadmap.
- Archived plans and status reports under `docs/archive/**` may preserve old roadmap language. They remain archive-only and are not active instructions.
- The Post-P6 prompt itself contains forbidden terms such as "optional geometry" and "truth" in audit/search instructions and forbidden-language examples. Those occurrences are intentional and are not active semantic claims.

## 9. Test And Quality-Check Results

Commands run locally:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest tests/training/test_trainer.py tests/operator/test_cli_operator.py
uv run pytest
```

Results:

- `uv run ruff check .` passed.
- `uv run ruff format --check .` passed: 284 files already formatted.
- `uv run mypy src` passed: no issues found in 176 source files.
- Targeted quick-win tests passed: 4 passed, 1 PyTorch nested-tensor warning.
- Full test suite passed: 195 passed, 1 PyTorch nested-tensor warning.

The warning is from PyTorch's prototype nested tensor API inside transformer modules. It is not caused by this pass and is non-blocking.

## 10. Remaining Known Risks

- No benchmark or scale-up result exists. P6 smoke and mini configs are runtime checks, not performance evidence.
- CI was not observed remotely in this pass. Local commands mirror `.github/workflows/ci.yml`.
- The local bare `uv run axiom ...` console script can require a synced/editable environment or `PYTHONPATH=src` in this developer checkout. CI installs the package with `pip install -e '.[dev]'`, and CLI behavior is covered through Typer tests.
- P6 training currently uses local smoke-scale defaults; GPU-scale training readiness still requires owner-selected environment and budget decisions.

## 11. Blockers, If Any

None.

## 12. Confirmation Checklist

- [x] Axiom remains a structured-native LLM training framework
- [x] Text projection remains mandatory and secondary
- [x] Geometry is native, not decorative; ablatable, not optional
- [x] AXF / AXT / AXC-out remain primary interfaces
- [x] No truth-label contamination
- [x] Time and lateral context remain distinct
- [x] Provider ingress remains data-construction only
- [x] Provider cache/replay/provenance preserved
- [x] Raw AXC-out / validated AXC-out / interpreted projection remain distinct
- [x] No hidden interpreter oracle repairs introduced
- [x] Loss masks and target availability semantics preserved
- [x] Negative sampling semantics preserved
- [x] Fairness rule is content/cutoff/split/params/compute-based
- [x] Token parity limited to text arms/projection
- [x] Geometry-off and non-geometric controls preserved
- [x] Context/provider shuffle controls preserved
- [x] No HKR proof overclaim introduced
- [x] README / CONCEPT / AGENTS / ROADMAP / ADRs aligned
- [x] CI or local quality checks run or limitations documented

## 13. Recommended Next Action

Review this audit and choose the next research-readiness or external-review work item. A practical next step is to prepare an external-review/GPU-sponsor readiness package from the post-P6 repository state without starting new model/training scope until the owner selects it.

## PR Body Draft

```markdown
## Summary

Post-P6 hardening pass for Axiom after P5/P6 landed. Tightened active documentation around geometry-native language, added small reproducibility/operator quick wins, and recorded the pass in `docs/work/POST_P6_HARDENING_AUDIT.md`.

## What was reviewed

README, AGENTS, Concept/ADR docs, roadmap, specs, configs, P1-P6 implementation surfaces, tests, CI config, and active work docs.

## Bugs fixed

- Replaced active "optional geometry" wording with nullable/maskable native geometry language.
- Updated stale next-action docs after Post-P6.
- Snapshot key AXT input artifacts into P6 run directories.
- Persist standalone `axiom experiment compare` outputs.

## Quick wins included

- Input artifact snapshotting for P6 run manifests.
- Comparison JSON output persistence.
- Explicit geometry provider kind in configs.
- Focused tests for the above.

## Conceptual guardrails checked

Structured-native I/O, secondary text projection, AXF/AXT/AXC-out boundaries, no truth labels, distinct time/lateral context, provider data-construction-only policy, no hidden interpreter oracle, masked targets, negative sampling, fairness rules, gauge-invariant/ablatable geometry, geometry/context/provider controls, and no HKR proof claims.

## Tests / checks run

- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src`
- `uv run pytest tests/training/test_trainer.py tests/operator/test_cli_operator.py`
- `uv run pytest`

## Remaining risks

- No benchmark or scale-up performance claim exists.
- Remote CI was not observed in this pass.
- GPU-scale readiness still needs owner-selected environment and budget decisions.

## Not done intentionally

- Did not replace structured-native model with stock CausalLM.
- Did not make geometry an optional plugin.
- Did not collapse AXC-out into text generation.
- Did not use providers in evaluation.
- Did not add truth labels.
- Did not start new roadmap scope beyond Post-P6 hardening.

## Next recommended action

Prepare an external-review/GPU-sponsor readiness package from the post-P6 repository state, or select the next research-readiness task.
```
