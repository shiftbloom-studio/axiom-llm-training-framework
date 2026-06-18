# Research Protocol

This project is an experimental framework. It does not prove HKR, physics,
metaphysics, or any physical claim. It tests whether an HKR-inspired claim-field
representation improves predictive and epistemic behavior in controlled machine
learning experiments.

## Scientific Posture

The strongest credible claim this repository can support in later stages is:

> Under controlled compute, token budget, data, and evaluation conditions, a
> claim-field representation improves specific predictive or epistemic behaviors
> relative to strong baselines.

That claim must be earned by comparison against flat text, structured text, and
ablated capsule variants. Until then, the repository should make no performance
claims.

## Negative Results

Negative results are expected and valuable. They may show that:

- capsule extraction noise destroys the signal;
- side channels add no useful information;
- structured text explains the gain without capsule-specific fields;
- popularity or degree controls explain the result;
- temporal evaluation is contaminated;
- HKR/geometric observables are decorative;
- the claim-field hypothesis needs to be narrowed or rejected.

Each stage should be designed so that those outcomes are visible rather than
hidden by a single aggregate score.

## Step 1 Rules

Step 1 must:

- build a strict versioned data contract;
- validate provenance, relations, uncertainty, temporal cutoffs, quality, and
  lineage;
- prevent obvious target leakage at the object level;
- track dataset files with manifests and hashes;
- provide tests that fail on invalid records.

Step 1 must not:

- train models;
- call external LLM APIs;
- fabricate benchmark results;
- add heavyweight training dependencies;
- encode HKR as a required truth claim.
