# Falsification Harness

Status: Current | Updated: 2026-06-18 | See: [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)

The current falsification harness is a preparation layer. It creates reproducible artifacts, controls, audits, metrics, hashes, and manifests so future training/evaluation work can ask the Axiom question fairly.

It does not train the v1 model, run benchmark evaluations, use external LLMs, or produce a verdict.

## Purpose

Axiom should not win by accident because data was rewritten, sources were counted, future information leaked, or geometry encoded a popularity shortcut.

The harness prepares controls for:

- flat text;
- structured text;
- capsule text rendering;
- no-provenance ablation;
- no-relations ablation;
- no-context/no-side-channel variants;
- context shuffle;
- provenance shuffle;
- popularity/frequency controls;
- temporal leakage audits.

## Relationship to Structured-Native v1

The current harness is infrastructure. Future P9/P10 work must extend it to orchestrate structured-native arms that consume AXT and emit AXC-out.

Current text-rendered artifacts are comparison surfaces, not the native Axiom model boundary.

## Temporal Leakage

Predictor-visible source timestamps must not be later than the capsule valid cutoff.

Future-facing fields may exist only as target fields. Rendered predictor text and predictor-side tensors must not expose:

```text
future_summary
future_label
target_timestamp
target_only
ground_truth
answer_key
```

Any temporal leakage finding should block downstream training preparation until repaired.

## Truth-Label Audit

The harness flags forbidden truth-label fields. These tests intentionally mention spellings such as `truth`, `is_true`, `is_correct`, `factuality`, and `ground_truth` so the validator can reject them.

New schemas and docs should prefer claim-state, evaluation_reference, gold_reference, or human_verified_reference where appropriate.

## Output Artifacts

Runs write:

```text
artifacts/falsification/<run_id>/
  manifest.json
  audits/
  metrics/
  arms/
```

Canonical AXC-derived artifacts use `.axc` when transformed records remain valid AXC/claim-state records. Noncanonical derived artifacts may use `.jsonl` and must be marked `derived_noncanonical`.

## CLI

Implemented commands:

```bash
axiom falsify run examples/axf/v0_1/minimal_dataset.axp --output-dir artifacts --arms all --seed 13 --allow-all-without-split
axiom falsify audit examples/axf/v0_1/minimal_capsules.axc --json
axiom falsify report artifacts/falsification/<run_id>/manifest.json --output report.md
```

## Hand-Off

The harness provides manifests and controls for later training/evaluation programs. It does not decide whether structured-native Axiom works. That verdict requires trained models, AXC-out scoring, text-projection comparisons, and a decision report.
