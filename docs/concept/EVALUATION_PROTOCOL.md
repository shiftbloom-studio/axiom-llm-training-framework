# Evaluation Protocol

Status: Current | Updated: 2026-06-18 | See: [CONCEPT.md](CONCEPT.md), [DECISIONS.md](DECISIONS.md)

Evaluation is not complete. This document records the active evaluation contract that future P9/P10 work must implement without overclaiming results.

## Primary Question

Does structured-native claim-field training improve epistemic competence compared with flat text under matched source content, temporal cutoffs, splits, extraction substrate, parameter budgets, compute/FLOPs, and schedules where applicable?

## Fairness

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

Token parity applies only inside:

- flat text vs structured text text-rendered arms;
- text-rendered baselines;
- text-projection losses and metrics.

Do not use token parity as the sole fairness rule for structured-native arms.

## Effects to Separate

Evaluation must isolate:

1. structured input;
2. structured auxiliary supervision;
3. architecture change;
4. learned geometry.

The experiment design must not let one factor explain another by accident.

## Required Arm Families

Future experiments should include:

- flat text baseline;
- structured text baseline;
- structured auxiliary-supervision arm;
- structured-native geometry-off arm;
- structured-native geometry-on arm;
- context-shuffle control;
- provenance-shuffle or no-provenance control;
- no-relations control;
- popularity/degree/recency controls;
- reduced-channel ablations.

Special tokens belong to text-rendered arms and text projection. They are not the native Axiom substrate.

## Structured Metrics

The primary model output surface is AXC-out. Evaluation should score raw model emissions separately from interpreted outputs.

Structured metrics should cover:

- claim-state prediction;
- relation prediction with negative and hard-negative samples;
- support/contradiction/supersession discrimination;
- provenance recovery;
- evidence-span pointer quality;
- uncertainty and calibration;
- temporal stability;
- context transport where geometry is enabled;
- outdated-belief handling;
- source sensitivity and provider-context sensitivity;
- temporal leakage audits.

## Text Metrics

Text metrics remain mandatory for comparability:

- text-projection loss/perplexity;
- text-rendered arm comparisons;
- human-readable projection checks where appropriate.

Text metrics do not replace structured metrics.

## Interpreter Guardrail

Evaluation must report:

```text
raw_emission score
validated_axc_out score
interpreted_projection score
text_projection score
```

The interpreter must not hide malformed raw emissions, add missing evidence, or repair outputs using future information.

## Geometry Guardrail

Geometry results must be compared against:

- geometry-off controls;
- parameter-matched non-geometric controls where applicable;
- context-shuffle controls;
- popularity/degree controls.

Axiom tests predictive and epistemic utility. It does not prove HKR.

## Decision Report

A positive, negative, or mixed result is valid only after actual training/evaluation runs produce artifacts, metrics, and a decision report. Step 5 was not completed by the legacy repository foundation work.
