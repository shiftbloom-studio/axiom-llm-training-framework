# Project Knowledge

Status: Current | Updated: 2026-06-18 | See: [CONCEPT.md](CONCEPT.md)

Axiom tests whether structured claim-field training improves epistemic competence compared with flat text under fair controls.

The public project identity is Axiom. Historical source material may use HoloCapsule terminology; the internal package and some internal class names may still use `hcaps`/`HoloCapsule` for compatibility.

## Research Thesis

Axiom treats claims as structured, temporally scoped, provenance-aware claim-states rather than plain token spans.

The primary substrate preserves:

- claim identity;
- provenance;
- relations;
- temporal scope;
- lateral context;
- epistemic state;
- target availability and loss masks;
- optional gauge-invariant geometry.

Text is a projection and comparison interface, not the primary system boundary.

## Format Family

- AXF: Axiom Exchange Format.
- AXC: Axiom Capsule Stream.
- AXP: Axiom Package.
- AXT: Axiom Tensor Bundle.
- AXC-out: structured Axiom model emission.

## Redundancy Discipline

Independent redundancy is not popularity, citation count, mention count, or duplicate text frequency. It is a claim-field signal that must be ablated and controlled.

Do not hard-code redundancy as correctness.

## Geometry Discipline

Geometry is in-plan, experimental, ablatable, and gauge-invariant. Raw learned matrices are implementation artifacts, not canonical semantic fields.

Context-shuffle and parameter-matched controls must be able to falsify any geometry contribution.

## Provider Discipline

Provider-backed extraction may be used for data construction only. It must be cached, hashed, replayable, manifest-backed, config-driven, and provider-identity tracked.

Provider identity and slant are lateral context.

## Evaluation Posture

The project is falsification-first. It should make failure modes distinguishable:

- poor extraction;
- weak structured supervision;
- decorative structure;
- geometry not carrying real context signal;
- popularity confounds;
- temporal leakage;
- a structured-native hypothesis that does not beat flat text under fair controls.

Negative results are valid research outcomes.
