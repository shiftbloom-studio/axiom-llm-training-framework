# Data Contract

Status: Current | Updated: 2026-06-18 | See: [CONCEPT.md](CONCEPT.md), [../../spec/AXF_V1.md](../../spec/AXF_V1.md)

Axiom records claim-state capsules, not truth-labeled text examples.

The current internal Pydantic model is `HoloCapsule` in `src/hcaps/schema/capsule.py`. That name is legacy internal compatibility. Public documentation should use Claim-State Capsule, AXC, AXP, AXT, AXC-out, and Axiom.

## Active Interfaces

| Interface | Role |
|---|---|
| AXC | Canonical claim-state capsule stream |
| AXP | Dataset package with manifests, hashes, sources, splits, capsules, and reports |
| AXT | Compiled tensor bundle for structured-native model input and targets |
| AXC-out | Structured model emission format |
| text projection | Secondary LLM compatibility and comparison surface |

P1 corpus ingress also emits construction-side artifacts linked to AXC/AXP:
provider cache manifests, extraction traces, provider disagreements, source
registries, relation candidates, negative pools, gold-reference candidates,
license reports, quality reports, and context side streams. These are
construction metadata, not model output.

P2 locks the v1 contract in:

- [../../spec/AXF_V1.md](../../spec/AXF_V1.md)
- [../../spec/AXC_V1_CAPSULE_SCHEMA.md](../../spec/AXC_V1_CAPSULE_SCHEMA.md)
- [../../spec/AXP_V1_PACKAGE_LAYOUT.md](../../spec/AXP_V1_PACKAGE_LAYOUT.md)
- [../../spec/AXT_V1_TENSOR_BUNDLE.md](../../spec/AXT_V1_TENSOR_BUNDLE.md)
- [../../spec/AXC_OUT_V1_SCHEMA.md](../../spec/AXC_OUT_V1_SCHEMA.md)
- [../../spec/FIELD_REGISTRY_V1.md](../../spec/FIELD_REGISTRY_V1.md)
- [../../spec/VOCABULARY_REGISTRY_V1.md](../../spec/VOCABULARY_REGISTRY_V1.md)

## Claim-State Capsule Fields

Claim-state capsules must preserve:

- claim identity;
- surface forms and source spans;
- temporal scope and valid cutoff;
- lateral context;
- provenance;
- typed relations;
- epistemic state;
- nullable/maskable gauge-invariant geometry observables;
- training eligibility and target declarations;
- quality and lineage metadata.

## Temporal Rules

Every predictor-visible source timestamp must be at or before the capsule valid cutoff.

Future-facing fields may exist only as targets. They must not be rendered into predictor text or compiled into predictor-side tensors.

Forbidden predictor exposure includes:

```text
future_summary
future_label
target_timestamp
target_only
ground_truth
answer_key
```

The `ground_truth` spelling may appear only in legacy forbidden-field tests or narrow evaluation-reference discussions. Prefer `gold_reference`, `human_verified_reference`, or `evaluation_reference` in new docs.

## Epistemic Rules

AXF/AXC/AXC-out must not contain binary truth labels.

Forbidden schema fields include:

```text
truth
is_true
correct
is_correct
factuality
ground_truth
label_truth
proven_true
```

Axiom may record support, contestation, uncertainty, stability, provenance, contradiction, revision, and evaluation references. It must not encode metaphysical truth as a training shortcut.

## Relation and Negative-Sampling Rules

Relations are model inputs and prediction targets. The data contract must preserve an n-ary/hypergraph path and deterministic negative-sampling metadata for:

- relation negatives;
- provenance negatives;
- context negatives;
- temporal negatives;
- near-but-distinct claims;
- same-topic unrelated claims.

Missing relation/provenance/context labels require masks. They are not negative examples.

## AXT Requirements

AXT must include both input tensors and target/output tensors. It must include explicit loss masks and target availability concepts such as:

```text
loss_mask_text_projection
loss_mask_relation_prediction
loss_mask_provenance_recovery
loss_mask_stability_prediction
loss_mask_uncertainty_calibration
loss_mask_future_summary
loss_mask_temporal_prediction
loss_mask_geometry_observables
loss_mask_provider_context
loss_mask_negative_sampling
```

## AXC-out Requirements

AXC-out must preserve:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

Raw emissions are scored directly. The interpreter is deterministic and ablatable.

## Geometry Rules

Geometry is optional, experimental, and ablatable. Canonical semantic outputs may include only gauge-invariant observables:

- curvature score;
- transport inconsistency;
- holonomy norm;
- normalized trace;
- spectrum summary;
- context-lability score.

Raw connection matrices and arbitrary basis-dependent gauge parameters are not canonical AXF or AXC-out semantic fields.

## Provider Rules

Local or remote LLM-compatible providers may be used for data construction/substrate harvesting only. Provider output must be cached, hashed, replayable, provenance-tracked, manifest-backed, config-driven, and ablatable.

Provider identity and slant are lateral context. Record provider identifiers in construction metadata whenever provider-backed extraction is used.

Provider traces belong in construction/provider manifests or linked corpus
artifacts unless a future AXF version formalizes a stable AXC field. Do not
force raw provider payloads into AXC semantic fields.
