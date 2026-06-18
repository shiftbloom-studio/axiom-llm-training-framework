# Claim-Field Substrate Builder

Status: Current | Updated: 2026-06-18 | See: [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md), [../work/IMPLEMENTATION_ROADMAP.md](../work/IMPLEMENTATION_ROADMAP.md)

The claim-field substrate builder converts source documents into validated claim-state capsules. It is foundation infrastructure for AXF/AXC/AXP and future AXT compilation.

The current implementation includes a deterministic bootstrap plus P1 provider-aware corpus ingress. P1 extends the ingress layer into a replayable corpus construction path without starting model training.

## Current Builder Flow

```text
source files
  -> source documents
  -> chunks
  -> candidate claims
  -> claim families
  -> relation candidates
  -> validated claim capsules
  -> AXC stream and manifests
```

The current internal schema still uses legacy names such as `HoloCapsule`. Public docs and CLI-facing language should use Axiom, Claim-State Capsule, AXC, AXP, AXT, and AXC-out.

## Supported Current Input

The reader currently supports:

- `.txt`
- `.md`
- `.markdown`
- `.jsonl` source records with a `text` field
- `.pdf` through an optional `pypdf` path when `pdf_reader_enabled` is set

PDF input is otherwise skipped with a clear `pdf_reader_disabled` or `pdf_reader_unavailable` warning. OCR is not introduced in P1.

## Canonical Output

Public canonical capsule streams use `.axc`.

The current CLI also writes a legacy internal JSONL path because the existing store predates the AXC public suffix. Public examples should include `--axc-output` for the canonical artifact:

```bash
axiom build-substrate \
  --input tests/fixtures/source_docs \
  --output data/claim-field/capsules.internal.jsonl \
  --axc-output data/claim-field/capsules.axc \
  --manifest data/claim-field/build_manifest.json \
  --cutoff-date 2026-01-01
```

## Provider Ingress Policy

P1 implements local or remote LLM-compatible provider interfaces for data construction/substrate harvesting only.

Provider-backed extraction must be:

- cached;
- hashable;
- replayable;
- manifest-backed;
- config-driven;
- ablatable;
- provider-identity tracked.

Record:

```text
provider_id
provider_family
provider_mode
model_name
confidence
escalation_reason
merge_decision
disagreement_set
```

Deterministic extraction remains a baseline/fallback. It is not the only allowed ingress mode.

Provider identity and slant are lateral context. Provider output must feed comparison arms equally after the substrate is pinned.

Current P1 CLI:

```bash
axiom corpus build --config configs/corpus/ml_software_benchmarks/corpus.yaml
axiom corpus providers check --config configs/corpus/ml_software_benchmarks/ingress.yaml
axiom corpus cache inspect .cache/axiom/providers
axiom corpus gold export --input artifacts/corpus/ml_software_benchmarks/dataset.axp --output artifacts/corpus/ml_software_benchmarks/gold_candidates.jsonl
```

## Temporal Guard

A build-level cutoff may exclude post-cutoff sources:

```bash
axiom build-substrate \
  --input tests/fixtures/source_docs \
  --output data/claim-field/capsules.internal.jsonl \
  --axc-output data/claim-field/capsules.axc \
  --manifest data/claim-field/build_manifest.json \
  --cutoff-date 2026-01-01
```

Undated sources must be tracked explicitly. Future-facing fields may be stored only as targets and must not enter predictor-visible text or tensors.

## Current Limitations

- The bootstrap extractor is transparent and testable, but shallow.
- Relation candidates are heuristic and low-confidence.
- Epistemic state fields are explicit proxies, not truth labels.
- Provider-backed extraction is implemented for corpus construction, but provider quality is not an evaluation verdict.
- No v1 model, training loop, learned geometry runtime, or benchmark verdict is implemented here.
