# Project Knowledge

The HoloCapsule project tests whether large language model pretraining improves
when the unit of data is a structured claim-field object rather than a flat
document stream.

## Core Research Idea

Claim-field pretraining treats a claim as a re-identifiable knowledge object.
The training unit is a `HoloCapsule`, not an arbitrary chunk. A capsule preserves
the text surface while making claim identity, provenance, relations,
uncertainty, temporal state, context, and future training targets explicit.

This repository starts from the data contract because every later experiment
depends on whether the substrate faithfully separates:

- text surface from claim identity;
- evidence from popularity;
- independent redundancy from raw mention count;
- current epistemic status from historical status;
- relation labels from free-form annotation text;
- future-facing targets from model-visible inputs.

## HoloCapsule as Training Substrate

A HoloCapsule is intended to become the unit from which future collators emit
text, side channels, relation neighborhoods, provenance targets, epistemic
scalars, and objective masks. The storage layer is therefore part of the active
learning substrate. It is not just a file dump.

The initial storage formats are JSONL and Parquet because they are inspectable,
reproducible, and easy to hash. Later stages may add graph, vector, or tensor
layouts only after the contract proves stable.

## First-Class Information

The schema treats the following as first-class:

- provenance and source timestamps;
- typed relations such as `supports`, `contradicts`, and `supersedes`;
- uncertainty and bounded epistemic scalars;
- temporal cutoffs for leakage prevention;
- context fibers and community identifiers;
- quality and lineage metadata;
- optional experimental geometric summaries.

## Redundancy Discipline

Redundancy is represented as an independent-community effective count. It is not
raw popularity, citation count, view count, or duplicate text frequency. The
schema enforces this distinction by naming the redundancy measure explicitly and
rejecting basis descriptions that collapse it into raw count proxies.

No redundancy law is hard-coded into the repository. Later experiments must be
able to test, ablate, or discard any proposed relationship.

## HKR and Geometry

HKR-inspired geometric components are optional and ablatable. When present, they
must be reported as gauge-invariant summaries such as loop norm, trace summary,
spectrum summary, or curvature score. Raw gauge matrices are not schema fields.

The project does not assume geometry will help. Context-shuffle and
parameter-matched controls must be able to falsify that branch.

## Evaluation Posture

The project is falsification-first. It should preserve negative results and make
failure modes distinguishable: poor extraction, weak side-channel use, decorative
geometry, popularity confounds, temporal leakage, or a core hypothesis that does
not beat flat text under fair controls.
