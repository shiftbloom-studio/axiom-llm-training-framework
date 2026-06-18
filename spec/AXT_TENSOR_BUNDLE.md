# AXT v0.1 Tensor Bundle

AXT is the Axiom Tensor Bundle: the compiled model-facing tensor bridge for AXF datasets.

AXT is specified before the production compiler exists so P3 can implement against a stable contract. AXT is not a prompt format and not a token-only dataset. It must include structured input tensors and structured target/output tensors.

## Files

Reserved v0.1 artifacts:

```text
<name>.axt
<name>.axt.safetensors
```

The `.axt` file is a manifest. The `.axt.safetensors` file contains tensors when tensor compilation is implemented.

## Required Manifest Concepts

An AXT manifest must record:

- AXF/AXC/AXP source paths and hashes;
- tensor bundle version;
- tokenizer identity and hash when text tensors are present;
- split identity and split hashes;
- arm or model-interface profile;
- tensor names, dtypes, shapes, and semantic roles;
- deterministic seed;
- build configuration hash;
- negative-sampling configuration and hash;
- target availability summary;
- output hash.

## Input Tensor Families

AXT input tensors may include:

- token IDs and attention masks for text-rendered arms or text projection;
- capsule IDs and claim-family IDs;
- claim-state feature tensors;
- temporal feature tensors;
- lateral-context feature tensors;
- relation incidence or hypergraph tensors;
- relation-neighborhood tensors;
- provenance/source tensors;
- evidence-span pointer inputs;
- epistemic scalar tensors;
- side-channel tensors;
- nullable/maskable gauge-invariant geometry feature tensors.

## Target Tensor Families

AXT target/output tensors may include:

- text-projection labels;
- claim-state targets;
- relation targets;
- provenance recovery targets;
- evidence-span pointer targets;
- stability targets;
- uncertainty/calibration targets;
- future-summary or future-state targets;
- geometry-observable targets;
- context-transport targets;
- AXC-out structured-output target fields.

Missing targets are masked, not treated as negatives.

## Required Mask Concepts

AXT must reserve explicit masks for:

```text
loss_mask_text_projection
loss_mask_relation
loss_mask_provenance
loss_mask_evidence_span
loss_mask_stability
loss_mask_uncertainty
loss_mask_future_summary
loss_mask_geometry
loss_mask_context_transport
```

Temporal masks must prevent future-facing fields from entering predictor-side tensors.

## Negative Sampling Metadata

Relation, provenance, and context objectives require negative and hard-negative samples. AXT manifests must make these deterministic and replayable.

Required sampler metadata:

- relation negative sampler;
- provenance negative sampler;
- context negative sampler;
- temporal negative sampler;
- near-but-distinct claim sampler;
- same-topic unrelated sampler.

## Geometry Policy

AXT may carry geometry features or targets only as gauge-invariant observables. Raw connection matrices and arbitrary basis-dependent gauge parameters are not canonical semantic tensors.

## Relationship to AXC-out

AXT target tensors define what the structured decoder can learn to emit. AXC-out defines the structured model emission surface. P2/P7 must keep these contracts aligned:

```text
AXT target tensors -> structured decoder -> AXC-out -> interpreter -> text projection
```

## Non-Goals For v0.1

AXT v0.1 does not implement:

- model architecture;
- training loop;
- learned geometry runtime;
- AXC-out interpreter runtime;
- benchmark claims.

AXT must remain traceable back to AXC records and AXP manifests.
