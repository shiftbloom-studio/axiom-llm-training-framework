# Axiom

**A structured-native LLM training framework for claim-field data.**

[![CI](https://github.com/shiftbloom-studio/axiom-llm-training-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/shiftbloom-studio/axiom-llm-training-framework/actions/workflows/ci.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)
![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)

Axiom asks one research question: does a model trained on structured claims learn epistemic competence better than the same model trained on flat text? Both sides get the same source content, temporal cutoffs, splits, extraction substrate, parameter budget, compute and training schedule. Token parity is only enforced inside the text-rendered arms.

Instead of feeding a causal LM tokenized prose, Axiom trains on claim-field records: claim identity, provenance, relations, epistemic state, temporal scope, lateral context, provider traces, negative pools and a learned geometry over the claim graph. The model emits structured output (AXC-out). Text projection exists for baselines, comparison with ordinary LLMs and human-readable output, and it stays a secondary interface.

> **Research status.** Axiom is pre-release research code. The full pipeline runs end to end at smoke and mini scale, but this repository makes no benchmark or model-performance claim.

## What's in the box

- **Corpus ingress.** Deterministic, local (`llama-server`) and remote providers build a claim-field substrate from source documents, with replayable caches, trace and disagreement manifests, source registries and negative pools.
- **Formats.** AXF v1 with AXC capsule streams, AXP dataset packages and AXT tensor bundles, plus field and vocabulary registries, masks and negative sampling specs in [`spec/`](spec/).
- **AXT compiler.** Turns an AXP package into a safetensors bundle with inputs, targets, masks, relation neighbourhoods and geometry slots, and ships a runtime dataset and batch interface.
- **Model stack.** A torch-native structured encoder with relation, provenance, provider and context conditioning, an epistemic router, a structured decoder, the AXC-out emission boundary and a text projection head.
- **Learned geometry.** Skew connections, matrix-exponential transport, bounded loop sampling and gauge-invariant observables. Every geometry component can be switched off for ablations.
- **Experiment runtime.** Training, experiment suites with separable arms (flat text through structured-native with geometry), scoring, a verdict ladder, and `inspect` / `export` / `compare` commands for runs.
- **Operator UI (optional).** A small local web UI in [`ui/`](ui/README.md) that prepares the same YAML configs and watches runs.

Reviewers should start with [docs/REVIEWER_GUIDE.md](docs/REVIEWER_GUIDE.md). The current work item is tracked in [docs/work/IMPLEMENTATION_ROADMAP.md](docs/work/IMPLEMENTATION_ROADMAP.md).

## Format Family

| Format | Meaning | Role |
|---|---|---|
| AXF | Axiom Exchange Format | Public claim-field format family |
| AXC | Axiom Capsule Stream | Claim-state capsule stream |
| AXP | Axiom Package | Dataset package with manifests, hashes, splits, sources, capsules, and reports |
| AXT | Axiom Tensor Bundle | Model-facing tensor bridge with input and target tensors |
| AXC-out | Axiom structured emission | Native structured model output before text projection |

AXC uses the public `.axc` suffix. It may be encoded as newline-delimited canonical JSON internally, but `.axc` is the public artifact type.

## Core Rules

Axiom preserves these rules unless a future accepted ADR explicitly changes them:

- no binary truth labels;
- no temporal leakage;
- time and lateral context are distinct axes;
- redundancy is not popularity;
- geometry is native (not decorative), learned as a real module, ablatable (not optional), with nullable/masked fields and controls, reported only via gauge-invariant observables;
- raw gauge matrices are not canonical semantic outputs;
- missing targets require loss masks and are not negative examples;
- negative samples are required for relation, provenance, and context objectives;
- external or local LLM providers may be used only for data construction/substrate harvesting;
- external LLMs are never used in training, evaluation, scoring, or benchmark judging unless a future ADR isolates and authorizes that use;
- raw model emissions must be stored and scored separately from interpreter output.

## Installation

Target runtime:

```text
Python >=3.14,<3.15
```

uv is the recommended package manager (the project uses PEP 735 / uv.lock).

### Development setup (uv)

```bash
git clone https://github.com/shiftbloom-studio/axiom-llm-training-framework.git
cd axiom-llm-training-framework

uv python install 3.14
uv sync --python 3.14 --extra dev

./bin/axiom --help
```

The small `bin/axiom` wrapper is a portable launcher. From inside the tree you can run `./bin/axiom ...` (or add the `bin/` directory to your PATH) without activating the venv or invoking `uv` yourself. `uv run` remains useful for developer tools (`pytest`, `ruff`, `mypy`, etc.).

`./bin/axiom` is the most reliable way to invoke the CLI. It resolves the project directory explicitly, so it works from inside the tree or when `bin/` is on your PATH, and it uses an invocation resilient to uv/hatchling editable-install edge cases that occasionally leave the generated `.venv/bin/axiom` shim unable to import `hcaps`.

### Alternative (non-uv) editable install

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Fish users can run `source .venv/bin/activate.fish` instead.

### Day-to-day usage

The primary supported CLI path is the wrapper:

```bash
./bin/axiom --help
```

If you put `bin/` on your `PATH`, the bare name works from any directory:

```bash
axiom --help
```

### CLI troubleshooting: "No module named 'hcaps'"

If you see:

```
ModuleNotFoundError: No module named 'hcaps'
```

(from `uv run axiom`, bare `axiom`, or the `.venv/bin/axiom` shim), the project package install in the venv is in a partial state. This can happen due to interactions between `uv run` on console scripts, hatchling editable builds, and prior failed uninstalls that leave a `dist-info` without a `RECORD` (or no accompanying `.pth` that adds `src/`).

**Fix (pick one):**

```bash
# Preferred quick repair (also works for non-uv venvs)
uv pip install -e . --python .venv/bin/python

# Or force via sync
uv sync --python 3.14 --extra dev --reinstall-package axiom-llm-training-framework
```

Running `./bin/axiom --help` (or any subcommand) bypasses the generated shim by launching the editable project directly. If you specifically need the generated shims (`uv run axiom`, activated `axiom`, or `.venv/bin/axiom`) to work directly, run one of the repair commands above.

## CLI Examples

The examples below use the bare `axiom` name for readability.

Inside the checkout the reliable invocation is:

- `./bin/axiom ...` (most reliable)
- `axiom ...` (with `bin/` on your `PATH`)

Validate an AXC stream:

```bash
axiom format validate examples/axf/v0_1/minimal_capsules.axc
```

Build a local claim-field substrate and write a canonical AXC stream:

```bash
axiom build-substrate \
  --input tests/fixtures/source_docs \
  --output data/claim-field/capsules.internal.jsonl \
  --axc-output data/claim-field/capsules.axc \
  --manifest data/claim-field/build_manifest.json \
  --cutoff-date 2026-01-01
```

Build the tiny provider-aware ML/software benchmark corpus fixture:

```bash
axiom corpus build \
  --config examples/corpus/ml_software_benchmarks/configs/deterministic_only.yaml
```

Inspect provider config and cache state:

```bash
axiom corpus providers check \
  --config configs/corpus/ml_software_benchmarks/ingress.yaml

axiom corpus cache inspect .cache/axiom/providers
```

Export human-review candidates from an AXP package:

```bash
axiom corpus gold export \
  --input artifacts/examples/ml_software_benchmarks/dataset.axp \
  --output artifacts/examples/ml_software_benchmarks/gold_candidates.jsonl
```

Compile an AXP package into an AXT tensor bundle:

```bash
axiom axt compile \
  --input examples/axf/v0_1/minimal_dataset.axp \
  --output artifacts/axt/minimal.axt \
  --config configs/axt/compile_smoke.yaml \
  --allow-all-without-split
```

Inspect and validate an AXT bundle:

```bash
axiom axt inspect artifacts/axt/minimal.axt
axiom axt validate artifacts/axt/minimal.axt
axiom axt tensor artifacts/axt/minimal.axt --group text_projection
```

Inspect and smoke-test the P4 structured-native model stack:

```bash
axiom model config-summary configs/model/structured_native_smoke.yaml
axiom model count-params configs/model/structured_native_smoke.yaml
axiom model smoke-forward artifacts/axt/minimal.axt \
  --config configs/model/structured_native_smoke.yaml
```

Inspect and smoke-test the P5 geometry module:

```bash
axiom geometry inspect-axt artifacts/axt/minimal.axt
axiom geometry smoke artifacts/axt/minimal.axt \
  --config configs/geometry/geometry_learned_smoke.yaml
```

Create falsification-preparation artifacts:

```bash
axiom falsify run \
  examples/axf/v0_1/minimal_dataset.axp \
  --output-dir artifacts \
  --arms all \
  --seed 13 \
  --allow-all-without-split
```

Render a falsification readiness report:

```bash
axiom falsify report \
  artifacts/falsification/<run_id>/manifest.json \
  --output report.md
```

These commands prepare and inspect data artifacts. They do not train the v1 model and do not produce an evaluation verdict.

Run the P6 smoke experiment runtime:

```bash
axiom experiment run configs/experiments/smoke_suite.yaml
axiom verdict report runs/p6_smoke_suite \
  --output runs/p6_smoke_suite/verdict/report.md
axiom run inspect runs/p6_smoke_suite
```

The smoke suite is a runtime check. It writes checkpoints, metrics, predictions, fairness reports, scores, and a verdict report, but it is not a benchmark or scale-up result.

### Full pipeline (one command)

For a single guided end-to-end training run from data extraction through finished checkpoints, run:

```bash
axiom full
```

`axiom full` prompts only for the required operator choices: source directory, temporal cutoff, training size (`smoke`, `pilot`, or `standard`), extraction backend (`integrated` llama-server cascade or remote Perplexity), and the model/API key required by that backend. Integrated mode starts `llama-server` locally from the selected Hugging Face GGUF model as the primary extraction provider, enables the built-in provider cascade with Perplexity Sonar as remote escalation, uses both only for corpus construction, then shuts the local server down before AXT compilation, training, scoring, or verdict generation. Remote mode uses Perplexity Sonar as the primary construction provider. API keys are held in-process via `PERPLEXITY_API_KEY` and are not written to artifacts.

The command preserves the full structured-native path: provider-aware corpus ingress, AXT compilation, the complete P6 arm catalog, learned geometry and controls, AXC-out/text projection, scoring, verdicts, and reproducibility artifacts. The resulting `runs/<generated-name>/` contains trained checkpoints (`arms/*/checkpoints/latest.pt`) and is fully usable with `axiom run inspect`, `axiom run export`, and `axiom run list`.

## v1 Roadmap

Axiom v1 is organized as six implementation programs in the current roadmap. Validation, tests, documentation, fixtures, manifests, and quality gates are done criteria inside each program, not separate roadmap phases.

1. P1 Claim-Field Corpus & Provider Ingress
2. P2 AXF v1 Contract, AXT & AXC-out Specification
3. P3 AXT Compiler & Runtime Data Interface
4. P4 Structured-Native Model Stack
5. P5 Learned Geometry & Claim-Field Graph Dynamics
6. P6 Training, Experiment Orchestration & Research Runtime

Legacy Steps 1-4 are foundation/infrastructure. Step 3 is a lightweight training bridge, not the P3 AXT compiler/runtime data-interface program; P4 is the structured-native model program. Step 4 prepares falsification artifacts, not the P6 evaluation verdict. Step 5 is not complete unless real training/evaluation artifacts and a decision report exist.

## Fairness

Primary fairness is not token parity alone.

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

Token parity applies only within text-rendered arms and text-projection losses/metrics.

The v1 experiments must separate:

1. structured input;
2. structured auxiliary supervision;
3. architecture change;
4. learned geometry.

## Documentation

Start with:

- [docs/concept/CONCEPT.md](docs/concept/CONCEPT.md)
- [docs/concept/DECISIONS.md](docs/concept/DECISIONS.md)
- [docs/work/IMPLEMENTATION_ROADMAP.md](docs/work/IMPLEMENTATION_ROADMAP.md)
- [docs/work/P2_HANDOFF_TO_P3.md](docs/work/P2_HANDOFF_TO_P3.md)
- [docs/work/P3_AXT_COMPILER_HANDOFF_TO_P4.md](docs/work/P3_AXT_COMPILER_HANDOFF_TO_P4.md)
- [docs/MODEL_STACK.md](docs/MODEL_STACK.md)
- [docs/GEOMETRY_MODULE.md](docs/GEOMETRY_MODULE.md)
- [docs/TRAINING_RUNTIME.md](docs/TRAINING_RUNTIME.md)
- [docs/EXPERIMENT_ORCHESTRATOR.md](docs/EXPERIMENT_ORCHESTRATOR.md)
- [docs/SCORING_AND_VERDICT.md](docs/SCORING_AND_VERDICT.md)
- [docs/OPERATOR_RUNTIME.md](docs/OPERATOR_RUNTIME.md)
- [docs/P6_HANDOFF_FINAL.md](docs/P6_HANDOFF_FINAL.md)
- [docs/work/P4_HANDOFF_TO_P5.md](docs/work/P4_HANDOFF_TO_P5.md)
- [docs/work/P4_HANDOFF_TO_P6.md](docs/work/P4_HANDOFF_TO_P6.md)
- [docs/work/P5_HANDOFF_TO_P6.md](docs/work/P5_HANDOFF_TO_P6.md)
- [docs/concept/AXT_RUNTIME_DATA_INTERFACE.md](docs/concept/AXT_RUNTIME_DATA_INTERFACE.md)
- [spec/AXF_V1.md](spec/AXF_V1.md)
- [docs/README.md](docs/README.md)

The internal Python package is still named `hcaps` for legacy compatibility. Public documentation should use Axiom terminology.

## Contributing

Bug reports, reproductions and focused pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and checks, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for how we work together. Please report security issues privately as described in [SECURITY.md](SECURITY.md).

## License

Copyright (C) 2026 Shiftbloom Studio.

Axiom is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE) for the full text.

If you run a modified version of Axiom (including the operator UI) as a network service, the AGPL asks you to offer its source code to the users of that service.
