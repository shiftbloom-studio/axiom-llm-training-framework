# Axiom v1 Implementation Roadmap

Status: Current | Updated: 2026-06-18 | Authority: [../concept/CONCEPT.md](../concept/CONCEPT.md), [../concept/DECISIONS.md](../concept/DECISIONS.md)

This is the current roadmap for completing Axiom v1. It replaces the older six-plan roadmap and legacy Step vocabulary as the active completion path.

This roadmap is not an implementation plan. It lists dependency-ordered programs. Each program must include its own code, schemas, CLI wiring where relevant, configs, docs, tests, fixtures, manifests/hashes where relevant, and quality gates. Validation and documentation are done criteria inside each program, not separate roadmap phases.

## Current Baseline

Built foundation/infrastructure:

- strict claim-state schema and validation;
- local substrate builder;
- AXF/AXC/AXP format support;
- lightweight training bridge;
- falsification-preparation harness;
- conformance fixtures and smoke configs.

Not complete:

- production AXT compiler;
- AXC-out runtime;
- structured-native model;
- learned geometry core;
- multi-objective training runtime;
- model-based evaluation verdict.

No benchmark or model-performance claim is made.

## Step vs Program Distinction

Legacy Steps 1-4 are foundation/infrastructure. They are useful history, not the active v1 completion map.

- Legacy Step 3 = lightweight training bridge. It is not P3.
- Legacy Step 4 = falsification artifact preparation. It is not P9 or a verdict.
- Legacy Step 5 is not complete unless actual model training, evaluation metrics, and a decision report exist.

P1 begins only after the pre-P1 documentation alignment pass is complete.

## v1 Programs

### P1 Claim-Field Corpus & Provider Ingress

Establish the first credible claim-field corpus and provider-backed ingress layer for ML/software benchmark claims. Keep deterministic extraction as a baseline/fallback. Provider-backed construction must be cached, hashed, replayable, manifest-backed, config-driven, ablatable, and provider-identity tracked.

Depends on: current foundation.

### P2 AXF v1 Contract, AXT & AXC-out Specification

Finalize the v1 data/interface contract before runtime work: AXF/AXC/AXP compatibility, AXT input and target tensor schema, AXC-out schema, target availability masks, loss masks, temporal masks, negative-sampling metadata, geometry-observable schema, and raw/validated/interpreted output boundaries.

Depends on: P1 corpus requirements.

### P3 AXT Compiler & Runtime Data Interface

Implement the production compiler from AXF/AXP/AXC to AXT tensor bundles and the runtime dataset/collator interface. AXT must contain input tensors, target/output tensors, loss masks, split metadata, manifest hashes, and deterministic negative-sampling metadata.

Depends on: P2.

### P4 Structured Encoder & Relation/Hypergraph Conditioning

Implement the structured encoder and relation-neighborhood conditioning path. Preserve the n-ary/hypergraph path and keep structured input ablatable from structured supervision and geometry.

Depends on: P3.

### P5 Learned Geometry Core

Implement the learned geometry module as a real trained component with gauge-invariant observables only. Geometry must be ablatable, parameter-matched where relevant, and tested with context-shuffle controls. JAX may remain reference/precompute; in-loop learned geometry is torch-native unless a later ADR changes that.

Depends on: P4 interfaces.

### P6 Full-Complexity Core & Epistemic Router

Implement the full-complexity core and epistemic router/escalation mechanism. The router is first-class and should not be simplified away. It must remain separable from input structure, auxiliary supervision, and geometry.

Depends on: P4 and P5 interfaces.

### P7 Structured Decoder, AXC-out Interpreter & Text Projection

Implement the structured decoder that emits AXC-out, the deterministic interpreter, and the secondary text-projection head. Preserve `raw_emission`, `validated_axc_out`, `interpreted_projection`, and `text_projection`; raw emissions must remain visible and scoreable.

Depends on: P6.

### P8 Multi-Objective Training Runtime & Curriculum

Implement structured-primary training with explicit loss masks, target availability, negative sampling, curriculum controls, reproducible seeds, checkpoints, and secondary text-projection losses.

Depends on: P7.

### P9 Experiment Arm Orchestrator & Decision Runtime

Implement experiment orchestration across flat text, structured text, structured-native, geometry-off, geometry-on, context-shuffle, provenance/relation ablations, and popularity/recency controls. Enforce fairness by source content, temporal cutoffs, splits, extraction substrate, parameter budget, compute/FLOPs, and schedule. Token parity applies only inside text-rendered arms and text-projection comparisons.

Depends on: P8.

### P10 Open Research Runtime & Operator Surface

Provide the operator-facing runtime, reports, reproducible artifact packaging, public examples, and open research workflows needed to run, inspect, reproduce, and communicate experiments without overclaiming results.

Depends on: P9.

## Next Action

```text
Create Implementation Plan P1: Claim-Field Corpus & Provider Ingress.
```

Do not create detailed plans for P2-P10 before P1 is accepted unless the owner explicitly asks for that planning pass.
