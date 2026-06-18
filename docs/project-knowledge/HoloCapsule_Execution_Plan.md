# HoloCapsule Claim-Field Pretraining — 5-Step Execution Plan

Status: working plan for an all-in research implementation, not a demo.

## Principle

The first implementation target is not a larger model. The first implementation target is a hard data contract and training interface that allow us to test whether claim-field structured pretraining beats flat text under controlled conditions.

The project must remain falsification-first. A negative result is valuable if the implementation is rigorous enough to tell us what failed: the capsule schema, the extraction process, the side-channel training interface, the geometric/HKR layer, or the broader hypothesis.

## The five steps

| Step | Commit-level outcome | What is actually done |
|---:|---|---|
| 1 | Repository constitution and HoloCapsule contract | Create the production-grade Python repository, lock the architecture documents, define the canonical HoloCapsule schema, implement validation, storage interfaces, manifests, config loading, tests, linting, typing, and CI. |
| 2 | Claim-field substrate builder | Implement ingestion from real corpora into document records, claim candidates, claim families, relation candidates, provenance objects, temporal metadata, deduplication metadata, and reproducible dataset manifests. |
| 3 | Training bridge | Implement tokenizer integration, Dataset/DataLoader/Collator, side-channel tensorization, objective masks, baseline flat-text stream, capsule-text stream, and capsule-side-channel stream. |
| 4 | Falsification harness | Implement temporal splits, confound baselines, context-shuffle, degree/popularity controls, no-provenance/no-relation/no-context ablations, metrics, experiment registry, and reproducible reports. |
| 5 | First real experimental run | Train and evaluate the first small but real models under identical compute/token budgets and produce the first decision report: proceed, redesign, or kill specific hypotheses. |

## Non-negotiable constraints

1. No “demo pipeline” architecture. Test fixtures are allowed; demo logic is not.
2. Every data artifact must carry provenance and a manifest entry.
3. No future information may enter a pre-cutoff training example.
4. Redundancy must not mean raw popularity or raw citation count.
5. HKR/geometric terms must be optional, ablatable, and never hard-coded as success.
6. Report only gauge-invariant geometric quantities.
7. Every improvement claim must be checked against flat text, structured text, and capsule ablations.
8. Negative results must be preserved in the experiment log.

## Recommended repository layout

```text
holocapsule-pretraining/
  README.md
  pyproject.toml
  uv.lock or requirements.lock
  .gitignore
  .pre-commit-config.yaml
  .github/workflows/ci.yml

  docs/
    ARCHITECTURE.md
    PROJECT_KNOWLEDGE.md
    RESEARCH_PROTOCOL.md
    DATA_CONTRACT.md
    EVALUATION_PROTOCOL.md
    DECISIONS.md
    source-material/
      README.md
      HKR_ML_Approach_Chapter.pdf
      LLM-Training_Best_Practices.pdf
      deep-research-storage.md
      deep-research-pretraining.md
      synthesis-pipelines.md

  configs/
    schema/default.yaml
    data/local.yaml
    train/smoke.yaml
    eval/smoke.yaml

  src/hcaps/
    __init__.py
    schema/
      __init__.py
      capsule.py
      document.py
      manifest.py
      identifiers.py
      validators.py
    store/
      __init__.py
      base.py
      jsonl.py
      parquet.py
      manifest.py
    ingest/
      __init__.py
    tensorize/
      __init__.py
    train/
      __init__.py
    eval/
      __init__.py
    utils/
      __init__.py
      hashing.py
      time.py
      logging.py

  tests/
    fixtures/
      valid_capsules.jsonl
      invalid_capsules.jsonl
    test_schema_capsule.py
    test_store_jsonl.py
    test_manifest.py
    test_temporal_guards.py
```

## Step-1 acceptance criteria

Step 1 is complete only when the following commands succeed locally:

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
mypy src
pytest
```

Step 1 must create no training results and make no performance claims.
