# HKR To AXF Mapping

Status: Current | Updated: 2026-06-18 | See: [CONCEPT.md](CONCEPT.md)

This document records the project interpretation that maps HKR-inspired research
concepts into neutral AXF data fields. The source-material folder contains
`HKR_ML_Approach_Chapter.pdf`; this mapping summarizes operational design
implications rather than quoting source text.

## Operational Mapping

| HKR-derived concept | AXF field or contract |
|---|---|
| fact-state / claim-state | AXC Claim-State Capsule |
| re-identifiable fact core | `ids.claim_family_id` and `claim.canonical_text` |
| context | `context` section and `ids.context_id` |
| O proxy | `epistemic_state.ontology_compatibility` |
| E proxy | `epistemic_state.evidential_anchoring` |
| T proxy | `epistemic_state.transformation_pressure` |
| R proxy | `epistemic_state.independent_redundancy` |
| context transport / curvature | native (nullable/maskable) `geometry` section |
| gauge identifiability | `geometry.gauge_policy = gauge_invariant_observables_only` |
| structured tensor bridge | AXT input and target tensors |
| structured model emission | AXC-out with raw, validated, interpreted, and text-projection layers |
| falsification-first methodology | AXP splits, holdouts, controls, and ablation manifests |
| no truth labels | stabilization, revision, uncertainty, and provenance fields |

## Claim-State, Not Truth Label

AXF does not encode `truth: true` or `is_correct: true`. It records observed
claim states under time, context, provenance, evidence, redundancy,
contradiction, revision, and stabilization dynamics.

## Geometry Is Reserved

AXF and AXC-out reserve geometry for context-transport observables. Canonical
records allow gauge-invariant summaries only. Raw learned connection matrices may
exist later in experiment artifacts, but they are not stable semantic fields.

Axiom tests whether these observables have predictive or epistemic utility. It
does not prove HKR.

## Falsification Path

AXP packages reserve explicit places for temporal holdouts, context shuffles,
degree-preserving rewires, popularity/recency controls, relation ablations,
geometry-disabled ablations, and provenance-disabled ablations. These controls
are part of the data contract because the research program is evaluated by
surviving controls, not by asserting positive results.
