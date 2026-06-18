# Axiom / HoloCapsule — Status, Progress & Code Review

Date: 2026-06-18 · Reviewer: automated code review · Commit: `b6f71e9` (branch `main`)
Scope: status of the implementation against the source-material vision (`docs/source-material/`) and the current canonical docs (`docs/`). This review describes and recommends only — it does not change any concept or any file under `docs/` or `docs/source-material/`.

---

## 1. Verdict

The repository is a **clean, rigorously engineered data-substrate + falsification toolchain that deliberately stops short of any machine learning**. Against the project's own 5-step plan, **Steps 1–4 are substantially implemented and green; Step 5 (the actual training network and model-based evaluation) does not exist yet** — which is consistent with the plan, not a regression.

Quality gates pass on the required runtime (Python 3.14): **95 tests pass, `ruff check` clean, `mypy --strict` clean across 53 files**, the `axiom` CLI imports and runs, and the falsification harness executes end-to-end on the example dataset. The codebase is honest about its limits: the "intelligence" (claim/relation extraction, epistemic scores) is intentional rule-based bootstrap, and the heavy ML pieces are reserved-but-unbuilt.

> Correction worth flagging up front: the comma-form `except A, B:` clauses in `falsification/runner.py` (lines 466, 522) are **valid Python 3.14** (PEP 758, parenthesis-free except). They are *not* the removed Python-2 syntax. On Python ≤3.13 they raise `SyntaxError`, which can cause a false "the CLI is broken" finding if the code is checked on an older interpreter. Verified: on 3.14 the module compiles, imports, and runs.

---

## 2. What I verified, and how

All of the following were executed against a fresh **CPython 3.14.5** environment (the project requires `>=3.14`):

| Gate | Result |
|---|---|
| `python -m compileall src` | clean (exit 0) — no real syntax errors anywhere |
| `pytest -q` | **95 passed** in ~0.6 s |
| `ruff check .` | All checks passed |
| `ruff format --check .` | **exit 1 — 2 files would be reformatted** (would fail CI) |
| `mypy src` (strict) | Success: no issues in 53 source files |
| `import hcaps.cli`, `import hcaps.falsification.runner` | import OK on 3.14 |
| `axiom --help` | exit 0 |
| `axiom falsify run examples/axf/v0_1/minimal_dataset.axp …` | produced 12 arms + audits + metrics + manifest |
| `axiom falsify report …` | produced an 84-line Markdown readiness report |

The two formatting-drift files are `src/hcaps/falsification/audits.py` and `tests/falsification/test_audits_temporal_leakage.py` — cosmetic only, but the CI `ruff format --check` step is currently red.

Note on tooling: the project uses Python-3.14-only syntax (PEP 695 `type X = …` aliases in both `identifiers.py` modules; PEP 758 except clauses in `runner.py`). This is fine given `requires-python >=3.14`, but it hard-locks the toolchain to a very new interpreter and will mis-flag on older ones.

---

## 3. Progress against the 5-step execution plan

Source: `docs/source-material/HoloCapsule_Execution_Plan.md`.

| Step | Plan outcome | Status | Notes |
|---|---|---|---|
| 1 | Repo constitution + HoloCapsule contract | **Done** | Strict Pydantic v2 schema, JSONL+Parquet stores, manifests, hashing, tests, lint, types, CI. |
| 2 | Claim-field substrate builder | **Done (bootstrap)** | Ingest → chunk → extract → families → relations → capsules → manifest. Extraction is rule-based by design. |
| 3 | Training bridge | **Partial** | Renderers, baseline tokenizer, in-memory dataset, list collator, side channels, manifest. Produces padded integer-list batches — **no tensors / no AXT files**. |
| 4 | Falsification harness | **Done** | 12 arms, leakage/integrity audits, dataset diagnostics, deterministic controls, run/audit/report CLI. Torch-free by design; runs end-to-end. |
| 5 | First real experimental run (train + evaluate + decision report) | **Not started** | No model, no training loop, no model-based evaluation, no decision report. |

Beyond the 5 steps, the team has also built the **public AXF format family** (specs in `spec/`: AXC capsule stream, AXP package, AXT tensor bundle). AXC and AXP are implemented; AXT is specified only.

---

## 4. Subsystem-by-subsystem code review

Legend: ✅ finished · 🟡 partial/deferred · 🧪 intentional bootstrap-heuristic · ⛔ not implemented

| Module | Status | Summary | Key placeholders / simplifications |
|---|---|---|---|
| `schema/` | ✅ | Canonical internal `HoloCapsule` contract: strict (`extra="forbid"`), temporal-leakage validators, redundancy-≠-popularity guard, bounded epistemic scalars, geometry gauge-policy. | None substantive. A few declared id types unused. |
| `store/jsonl.py` | ✅ | Streaming canonical JSONL read/write/validate. | — |
| `store/parquet.py` | 🟡 | Works, but stores 5 flat columns + the whole capsule as a `record_json` **string**, not native nested Arrow. | Nested layout explicitly deferred (`parquet.py:1-8`). |
| `format/` (AXC) | ✅ | AXC capsule model + adapter, NDJSON streams, validation (incl. forbidden truth-field scan), identifiers, hashing, AXP root manifest. Spec-conformant. | Adapter hardcodes `leakage_validation_status=PASSED` and `status=UNASSESSED` (`format/capsule.py:358,365`). |
| `format/package.py` (AXP) | 🟡 | Create skeleton / build-from-stream / validate / inspect all work for the v0.1 JSONL profile. | Split files & several manifests written **empty** (`{"capsule_ids": []}`, `{}` at `package.py:78-81`); `contexts.parquet`/`provenance.parquet` not written; no `.axp.tar.zst`. |
| AXT tensor bundle | ⛔ | Only a version constant, two extension strings, and a `compiled/README.md` placeholder. | Spec-sanctioned deferral (`spec/AXT_TENSOR_BUNDLE.md:5-6`). |
| `ingest/readers.py` | 🟡 | Deterministic `.txt/.md/.markdown/.jsonl` readers, sidecar metadata, per-file hashing. | **PDF detected and skipped** (`readers.py:55-64`); UTF-8 only; license defaults to `"unknown"`. |
| `ingest/chunking.py` | ✅ | Heading → paragraph → sentence-packing chunker with char offsets. | Markdown-heading + punctuation-regex sentence split (naive on abbreviations/decimals). |
| `extraction/claims.py` | 🧪 | "Claims" = sentences containing hardcoded assertive cues (" is ", " shows ", …); type via keyword table; confidence is a hand-tuned formula. No model/LLM. | Self-labelled bootstrap; `ClaimExtractor` Protocol seam reserved for a future neural/LLM extractor. |
| `extraction/relations.py` | 🧪 | Relations = cue words + lexical token overlap (Jaccard), all-pairs O(n²). Only 6 of 12 relation types can ever be emitted; direction is unverified. | Self-labelled bootstrap; hand-set thresholds. |
| `substrate/builder.py` | ✅ orchestration / 🧪 values | Full pipeline orchestration, temporal-cutoff filtering (real strength), JSONL+AXC(+AXP) output, reproducibility manifest. | Several capsule epistemic fields are constants/proxies: `ontic_compatibility=0.5`, `transformation_pressure=0.0`, `stability_label=UNKNOWN`, "independent sources" = distinct-document count (`builder.py:276-289,357`). |
| `substrate/canonicalize.py` | ✅ | Lexical similarity (token Jaccard + difflib) + stable SHA ids. | Lexical only — no embeddings; English stopwords; drives family clustering. |
| `training_bridge/tokenizer.py` | 🧪 | `WhitespaceTokenizer` = `text.lower().split()`. | Baseline only; `TokenizerProtocol` seam for a real BPE/HF tokenizer. |
| `training_bridge/dataset.py` | 🟡 | Reads whole AXC/AXP into memory, yields tokenized dicts; labels = copy of input_ids (naive next-token). | In-memory only; returns Python lists, not tensors. |
| `training_bridge/collator.py` | 🟡 | Manual list padding, `-100` label pad, stacks side channels. | Returns dict of lists; no tensor/safetensors output. |
| `training_bridge/examples.py` | ✅ | `flat_text` / `structured_text` / `capsule_text` renderers; excludes future targets. | — |
| `training_bridge/side_channels.py` | ✅ | 8 numeric features from epistemic/relation/provenance/geometry. | Fixed feature list; **never consumed by any model** (no adapter exists). |
| `falsification/` | ✅ | 12 arms (baselines, ablations, shuffles, popularity & temporal controls), leakage/integrity audits, deterministic controls, run/audit/report. Runs end-to-end. | `metrics.py` computes **dataset diagnostics, not model-quality scores** (by design). |
| `cli.py` | ✅ | `build-substrate`, `inspect-substrate`, `format validate`, `package init/validate/inspect`, `falsify run/audit/report`. | No `train` / `tensorize` / `evaluate` commands. |

---

## 5. Inventory of mock / placeholder / heuristic / example content

These are the parts that look like data/behaviour but are intentionally provisional. None is hidden — most are documented in `docs/CLAIM_FIELD_SUBSTRATE_BUILDER.md` ("Current Limitations") and the specs.

- **Bootstrap claim extraction** — keyword/regex cues, not understanding (`extraction/claims.py`).
- **Bootstrap relation detection** — cue words + bag-of-words overlap; unverified direction; 6/12 relation types unreachable (`extraction/relations.py`).
- **Hardcoded epistemic proxies** in assembled capsules — `ontic_compatibility`, `transformation_pressure`, `stability_label`, independent-source count (`substrate/builder.py`).
- **Whitespace tokenizer** standing in for a real subword tokenizer (`training_bridge/tokenizer.py`).
- **Parquet `record_json`** standing in for native nested Arrow (`store/parquet.py`).
- **AXP empty placeholders** — `splits/*.json`, `manifests/{construction,leakage,deduplication,licenses,confound_controls}.json` are created empty; control *locations* are reserved but content isn't generated (`format/package.py`).
- **AXT** — specified, not implemented.
- **Reserved compression** — `.axc.zst`, `.axp.tar.zst` are extension constants only.
- **Example dataset** — `examples/axf/v0_1/` (valid + intentionally-invalid capsules, a minimal `.axp` package); this is real, validates, and is used by tests.
- **Disabled configs** — `configs/train/smoke.yaml` (`enabled: false`, references a not-yet-existing `hcaps.tensorize` entrypoint) and `configs/eval/smoke.yaml` (`enabled: false`).

---

## 6. What's missing for a full-scale, complete training network/tool

The current repo ends at "validated, falsification-ready training streams." To become a complete training tool that can actually answer the project's hypothesis, the following are the remaining build-out areas. These map directly onto the existing plan (WP4–WP9 in `HoloCapsule_Claim_Field_Pretraining_Architecture.md`, Step 5 in the execution plan) and the architecture's own flow diagram — they are not new design decisions.

1. **Tensorization → AXT** (the "tensorizer" box). Turn the in-memory padded batches into persisted `.axt` / `.axt.safetensors` bundles (token ids, masks, loss masks, side-channel tensors, relation/provenance tensors, split metadata, source manifest hashes) traceable back to AXC/AXP. Needs a numeric/array dependency (numpy/torch/safetensors) — currently none are in `pyproject.toml`.
2. **A real tokenizer.** Replace the whitespace baseline with BPE/WordPiece/HF behind the existing `TokenizerProtocol`, with persisted vocab/merges and stable special-token ids.
3. **The training network itself** (the "training network" box; WP5). Model definitions for the comparison arms — Baseline A (flat-text LM), B (structured-text), C (capsule-serialized), D (capsule + side-channel adapter) — plus the training loop: optimizer, scheduler, loss/objectives, loss masking, seed control, checkpointing. No model code exists anywhere (no torch/jax/tf in the project).
4. **Capsule adapter / side-channel + feature encoder** (WP6). The 8-D side channels and relation neighbourhoods are produced but never consumed; a prefix/adapter/router that injects them into the backbone is needed to test arms C/D.
5. **Model-based evaluation harness** (WP7). `docs/EVALUATION_PROTOCOL.md` lists the targets — validation/next-token loss under equal token budget, relation prediction, support/contradiction classification, provenance recovery, uncertainty calibration, temporal-stability prediction, outdated-belief handling. Today's `falsification/metrics.py` only counts dataset properties; nothing scores a model.
6. **Experiment / ablation runner that trains and compares arms** under matched token/parameter/compute budgets, then emits the **decision report** (proceed / redesign / kill). The falsification harness *prepares* comparable arms; it does not *train or score* them.
7. **Synthetic surface views** (FAQ, teaching note, counterargument, historical update) referenced in the architecture and AXC `surface_forms` — slots exist; generation does not.
8. **Optional claim-field graph signals** (WP9) — relation-neighborhood sampler / community-diversity / conflict-degree / later HKR curvature proxies. Schema reserves geometry; no computation exists.
9. **Upgraded substrate intelligence** to feed real training: neural/LLM claim & relation extraction (Protocol seams already in place), genuine source-independence estimation, evidence-type classification, and scalable dedup (embeddings + ANN/blocking) instead of greedy lexical clustering.
10. **Scale & ops**: streaming/sharded dataset I/O (currently fully in-memory), PDF and broader readers, and experiment tracking (W&B/MLflow/TensorBoard). The dependency stack (Step-1 deliberately torch-free) will need to grow for any of the above.

---

## 7. Smaller issues & observations (non-blocking)

- **CI format gate is red**: `ruff format --check` fails on `falsification/audits.py` and `tests/falsification/test_audits_temporal_leakage.py`. One `ruff format` run fixes it.
- **Missing `src/hcaps/training_bridge/__init__.py`** (every other subpackage has one). Works today via implicit namespace packages, but `import hcaps.training_bridge` exposes nothing and it's inconsistent.
- **Deprecated `datetime.utcnow()`** at `training_bridge/manifests.py:66` (the falsification manifest correctly uses `datetime.now(UTC)`).
- **Doc/code drift**: `docs/ARCHITECTURE.md` and `docs/DECISIONS.md` (ADR-0004) still describe the "Step 1 boundary" as current ("does not own … tokenizer integration"), but the code has since added the substrate builder, AXF formats, the training bridge (with a tokenizer), and the falsification harness. The narrative docs lag the code. (Flagging only — not edited per your instruction.)
- **Untracked duplicate files at repo root**: the `docs/` and `docs/source-material/` files also exist as untracked copies in the repository root (e.g. `ARCHITECTURE.md`, `HKR_ML_Approach_Chapter.pdf`, `deep-research-report.md`), and `README.md` has a large uncommitted reduction. Worth a cleanup decision (yours to make).
- **Cleanup needed (my artifact)**: I created a temporary virtualenv `.uvtest/` in the repo to run the suite on Python 3.14. The sandbox can't delete it (the mounted filesystem rejected unlink) and it is **not** git-ignored. Please remove it with `rm -rf .uvtest` from your machine so it doesn't show as untracked files.

---

## 8. Bottom line

Steps 1–4 are real, tested, internally consistent, and faithful to the falsification-first contract: strict schema, temporal-leakage guards, provenance/manifests/hashing, a working AXF/AXC/AXP format layer, and a falsification harness that produces honest dataset diagnostics and comparable arms. The substrate's "intelligence" is deliberately a transparent bootstrap with clean Protocol seams for stronger backends. The single largest gap to a "full-scale complete training tool" is everything downstream of the training bridge — **tensorization to AXT, a real tokenizer, the model(s), the training loop, the side-channel adapter, and a model-based evaluation/decision harness (Step 5)** — none of which exists yet, by design. The next commit-sized step that unlocks the most is tensorization + a minimal trainable baseline (arms A/B) so the harness's prepared arms can finally be trained and scored.
