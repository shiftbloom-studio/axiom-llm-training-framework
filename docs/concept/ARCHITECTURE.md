# Architecture

Status: Current | Updated: 2026-06-18 | See: [CONCEPT.md](CONCEPT.md), [DECISIONS.md](DECISIONS.md), [../work/IMPLEMENTATION_ROADMAP.md](../work/IMPLEMENTATION_ROADMAP.md)

Axiom is a structured-native LLM training framework. The active architecture preserves language compatibility, but it does not define the system boundary as plain text.

## Target Flow

```text
source material
  -> claim-field corpus and provider ingress
  -> AXF / AXC / AXP
  -> AXT structured tensor bundle
  -> structured encoder
  -> relation / hypergraph conditioning
  -> learned geometry core
  -> full-complexity core
  -> epistemic router
  -> structured decoder
  -> AXC-out
  -> deterministic interpreter
  -> text projection
```

Text projection is mandatory for LLM comparability. It is not the primary substrate.

## Current Repository Layers

Implemented foundation:

- strict claim-state schema and validators;
- local source ingestion and substrate builder;
- AXC streams and AXP package helpers;
- lightweight text-rendering training bridge;
- falsification-preparation harness and audits;
- conformance fixtures.

Reserved or future v1 layers:

- production AXT compiler;
- AXT input and target tensors;
- structured-native model;
- learned geometry module;
- epistemic router;
- structured decoder;
- AXC-out interpreter;
- training runtime;
- experiment arm orchestrator and decision reports.

## AXT Boundary

AXT is the bridge from AXF records to model-facing tensors. It must carry:

- predictor-side input tensors;
- target/output tensors;
- loss masks and target availability;
- temporal masks;
- relation/provenance/context negative-sampling metadata;
- source and manifest hashes;
- deterministic seeds and build configuration references.

AXT is not a text prompt format. Token tensors may exist for baselines and text projection, but structured tensors remain first-class.

## AXC-out Boundary

AXC-out is the structured model emission format. It must preserve:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

The interpreter may validate and project. It may not hide raw model failures or add oracle knowledge.

## Temporal and Context Axes

Time and lateral context are separate:

- temporal axis: source date, valid cutoff, observed-at time, target time, drift;
- lateral context: provider, community, field, source, method, relation neighborhood.

Predictor-side timestamps must be at or before the valid cutoff. Future-facing fields may exist only as masked targets.

## Geometry

Geometry is in-plan, experimental, and ablatable. Only gauge-invariant observables are canonical outputs. Raw learned matrices may exist as experiment artifacts, not semantic AXF/AXC-out fields.

## Legacy Step Boundary

Older architecture notes used Step 1-5. Those steps are historical infrastructure milestones:

- Step 3 is a lightweight training bridge, not the structured-native P3 model program.
- Step 4 prepares falsification artifacts, not the evaluation verdict.
- Step 5 is not complete without actual training/evaluation artifacts and a decision report.
