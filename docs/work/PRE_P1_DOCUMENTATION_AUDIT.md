# Pre-P1 Documentation Audit

Date: 2026-06-18

## 1. Summary

Completed a pre-P1 documentation and metadata alignment pass for Axiom as a structured-native LLM training framework. The active documentation now uses Axiom terminology, points to the P1-P10 roadmap, preserves AXC/AXT/AXC-out as structured interfaces, and records P1 as the next implementation action.

No P1 runtime work, model work, training work, geometry runtime work, or benchmark/evaluation claim was started.

## 2. Files Inspected

Inspected root documentation and metadata:

- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `LICENSE`
- `pyproject.toml`
- `uv.lock`
- `.python-version`
- `.github/workflows/ci.yml`

Inspected active docs:

- `docs/README.md`
- `docs/concept/CONCEPT.md`
- `docs/concept/DECISIONS.md`
- `docs/concept/ARCHITECTURE.md`
- `docs/concept/DATA_CONTRACT.md`
- `docs/concept/EVALUATION_PROTOCOL.md`
- `docs/concept/FALSIFICATION_HARNESS.md`
- `docs/concept/HKR_TO_AXF_MAPPING.md`
- `docs/concept/PROJECT_KNOWLEDGE.md`
- `docs/concept/RESEARCH_PROTOCOL.md`
- `docs/concept/TRAINING_BRIDGE.md`
- `docs/concept/CLAIM_FIELD_SUBSTRATE_BUILDER.md`
- `docs/work/IMPLEMENTATION_ROADMAP.md`
- existing legacy work/status docs before archiving

Inspected specs, examples, configs, and public-facing code strings:

- `spec/AXF.md`
- `spec/AXC_CAPSULE_SCHEMA.md`
- `spec/AXP_PACKAGE_LAYOUT.md`
- `spec/AXT_TENSOR_BUNDLE.md`
- `examples/axf/v0_1/**/README.md`
- `configs/**/*.yaml`
- `src/hcaps/**/__init__.py`
- `src/hcaps/cli.py`
- selected public docstrings/messages in `src/hcaps/schema`, `src/hcaps/store`, and `src/hcaps/falsification`

Inspected historical material:

- `docs/project-knowledge/**`

## 3. Files Updated

Canonical and navigation docs:

- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/README.md`
- `docs/concept/CONCEPT.md`
- `docs/concept/DECISIONS.md`
- `docs/work/IMPLEMENTATION_ROADMAP.md`

Protocol/concept docs:

- `docs/concept/ARCHITECTURE.md`
- `docs/concept/CLAIM_FIELD_SUBSTRATE_BUILDER.md`
- `docs/concept/DATA_CONTRACT.md`
- `docs/concept/EVALUATION_PROTOCOL.md`
- `docs/concept/FALSIFICATION_HARNESS.md`
- `docs/concept/HKR_TO_AXF_MAPPING.md`
- `docs/concept/PROJECT_KNOWLEDGE.md`
- `docs/concept/RESEARCH_PROTOCOL.md`
- `docs/concept/TRAINING_BRIDGE.md`

Specs, examples, configs, and metadata:

- `spec/AXF.md`
- `spec/AXC_CAPSULE_SCHEMA.md`
- `spec/AXP_PACKAGE_LAYOUT.md`
- `spec/AXT_TENSOR_BUNDLE.md`
- `spec/AXC_OUT_SCHEMA.md`
- `examples/axf/v0_1/README.md`
- `examples/axf/v0_1/minimal_dataset.axp/compiled/README.md`
- `configs/eval/smoke.yaml`
- `configs/schema/default.yaml`
- `configs/train/smoke.yaml`
- `pyproject.toml`
- `uv.lock`
- `docs/project-knowledge/README.md`

Tiny public-facing code string/docstring updates:

- `src/hcaps/__init__.py`
- `src/hcaps/schema/__init__.py`
- `src/hcaps/schema/capsule.py`
- `src/hcaps/schema/identifiers.py`
- `src/hcaps/schema/manifest.py`
- `src/hcaps/schema/validators.py`
- `src/hcaps/store/__init__.py`
- `src/hcaps/store/base.py`
- `src/hcaps/store/jsonl.py`
- `src/hcaps/store/parquet.py`
- `src/hcaps/utils/__init__.py`
- `src/hcaps/falsification/controls.py`
- `src/hcaps/falsification/reports.py`

## 4. Files Archived or Deprecated

Archived/deprecated:

- `docs/archive/legacy-plans/PLAN_1_substrate.md`
- `docs/archive/legacy-plans/PLAN_2_data_interface.md`
- `docs/archive/legacy-plans/build_map_legacy_six_plan.svg`
- `docs/archive/status/PROGRESS_REVIEW_2026-06-18.md`
- `docs/archive/README.md`

Removed disposable artifact:

- `.reorg_probe2.txt`

## 5. Remaining Non-Blocking Issues

- The internal Python package remains `hcaps`, and the internal model class remains `HoloCapsule`. This is intentional legacy compatibility and is now documented.
- The current substrate CLI still writes a legacy internal JSONL path in addition to canonical AXC output. Public examples now show `--axc-output`.
- Historical files under `docs/project-knowledge/` and archived files under `docs/archive/` retain old terminology and may contain stale links. They are explicitly marked historical/superseded.
- No `CITATION.cff` was added because there is no DOI or publication metadata to cite without inventing data.

## 6. Blocking Issues

None.

## 7. Confirmation Checklist

- [x] structured-native LLM framing aligned
- [x] old HoloCapsule public naming removed or contained
- [x] fairness rule updated
- [x] external LLM provider policy updated
- [x] AXC-out referenced
- [x] AXT referenced
- [x] text projection preserved
- [x] no truth labels preserved
- [x] geometry gauge-invariant / ablatable
- [x] Step vs Plan distinction clear
- [x] P1 is the next implementation program

## 8. Checks Run

Documentation/metadata grep checks:

- searched active docs/specs/examples/configs/source metadata for stale project naming, old six-plan wording, blanket external-LLM bans, `.axc.jsonl` public examples, token-only collapse language, and geometry overclaims;
- remaining matches are explicit guardrails, legacy internal API references, code compatibility constants/tests, or archived/historical material.

Quality gates:

```bash
uv lock
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pytest
uv run ruff check .
```

Results:

- `uv lock`: succeeded; package renamed in lockfile.
- `uv run ruff check .`: passed.
- `uv run ruff format .`: succeeded; 2 files reformatted.
- `uv run mypy src`: passed, 53 source files.
- `uv run pytest`: passed, 95 tests.
- final `uv run ruff check .`: passed.

## 9. Exact Next Action

Create Implementation Plan P1: Claim-Field Corpus & Provider Ingress.
