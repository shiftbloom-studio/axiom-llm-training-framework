# HoloCapsule Claim-Field Pretraining

This repository is the research foundation for HoloCapsule Claim-Field
Pretraining. Step 1 creates the canonical schema, validation layer, storage
interfaces, manifests, documentation, tests, and CI. It does not train a model,
call external LLM APIs, or report benchmark results.

The central unit is a validated `HoloCapsule`: a versioned claim-field record
that keeps claim identity, text surfaces, provenance, temporal state, relations,
epistemic signals, context, training targets, quality controls, and lineage in
one reproducible object.

## Requirements

- Python 3.14+
- `uv` for Python version management and local environments

```bash
uv python install 3.14
uv venv --python 3.14 --seed venv
source venv/bin/activate
python -m pip install -e '.[dev]'
```

## Quality Gates

```bash
ruff check .
ruff format --check .
mypy src
pytest --cov=hcaps
```

## Repository Shape

- `src/hcaps/schema/` contains the canonical Pydantic v2 data contract.
- `src/hcaps/store/` contains JSONL, Parquet, and manifest storage helpers.
- `configs/` contains smoke configuration placeholders for later stages.
- `docs/` contains the architecture, research protocol, data contract, future
  evaluation protocol, ADRs, and source-material policy.
- `tests/` contains fixture-driven validation tests.

Training, tensorization, substrate building, and evaluation harnesses are later
steps. This step deliberately keeps heavyweight training dependencies out of the
runtime graph.
