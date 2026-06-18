# Axiom

**Axiom is a structured-native LLM training framework.**

Axiom tests whether claim-field structure improves epistemic competence compared with flat text under matched source content, temporal cutoffs, splits, extraction substrate, parameter budgets, compute/FLOPs, and training schedules where applicable.

The project remains LLM-compatible through text projection, token baselines, text-rendered arms, and secondary text losses. Its primary substrate is not a flat token stream. Its primary substrate is structured claim-field data: claim identity, provenance, relations, epistemic state, temporal scope, lateral context, and optional gauge-invariant geometry.

Text is a projection and comparison interface. AXC, AXT, and AXC-out are the primary structured interfaces.

## Status

Current repository status:

- Foundation, substrate, AXF/AXC/AXP format support, provider-aware P1 corpus ingress, a lightweight training bridge, and falsification-preparation artifacts exist.
- P1 adds deterministic/local/remote-compatible data-construction providers, replayable provider caches, trace/disagreement manifests, source registries, negative pools, gold-reference candidate hooks, and a tiny synthetic ML/software benchmark corpus fixture.
- P2 defines AXF v1, AXC v1, AXP v1, AXT v1, AXC-out v1, field/vocabulary registries, masks, negative sampling, provider traces, and interpreter boundaries.
- AXT is specified as the tensor bridge, but the production AXT compiler is not complete.
- AXC-out is specified as the structured model emission format, but the runtime decoder/interpreter is not complete.
- The v1 structured-native model, learned geometry module, multi-objective training runtime, and evaluation verdict are not complete.
- No benchmark or model-performance claim is made by this repository.

The next implementation action is:

```text
Create Implementation Plan P3: AXT Compiler & Runtime Data Interface.
```

P1 and P2 are substrate/interface work only. They do not train the model or produce an evaluation verdict.

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
- geometry is experimental, ablatable, and gauge-invariant;
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

Development setup:

```bash
git clone https://github.com/shiftbloom-studio/axiom-llm-training-framework.git
cd axiom-llm-training-framework
uv sync --python 3.14
uv run pytest
```

Alternative editable install:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## CLI Examples

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

## v1 Roadmap

Axiom v1 is organized as six implementation programs in the current roadmap. Validation, tests, documentation, fixtures, manifests, and quality gates are done criteria inside each program, not separate roadmap phases.

1. P1 Claim-Field Corpus & Provider Ingress
2. P2 AXF v1 Contract, AXT & AXC-out Specification
3. P3 AXT Compiler & Runtime Data Interface
4. P4 Structured-Native Model Stack
5. P5 Learned Geometry & Claim-Field Graph Dynamics
6. P6 Training, Experiment Orchestration & Research Runtime

Legacy Steps 1-4 are foundation/infrastructure. Step 3 is a lightweight training bridge, not the P3 structured-native model program. Step 4 prepares falsification artifacts, not the P9/P10 evaluation verdict. Step 5 is not complete unless real training/evaluation artifacts and a decision report exist.

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
- [spec/AXF_V1.md](spec/AXF_V1.md)
- [docs/README.md](docs/README.md)

The internal Python package is still named `hcaps` for legacy compatibility. Public documentation should use Axiom terminology.

## License

Apache-2.0. See [LICENSE](LICENSE).
