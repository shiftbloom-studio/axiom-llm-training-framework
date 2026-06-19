# AXC v0.1 Capsule Schema

AXC is a newline-delimited stream of claim-state capsule records. Each line is a
canonical JSON object with:

```text
format = "AXC"
format_version = "0.1.0"
```

## Required Sections

- `ids`: capsule, claim family, claim state, and context identifiers.
- `claim`: canonical claim identity and type.
- `surface_forms`: canonical expression, source spans, normalized views, and
  generated views.
- `temporal`: valid-as-of date, construction timestamp, cutoff policy, and
  leakage validation status.
- `context`: domains, communities, and optional context description.
- `epistemic_state`: stabilization status and epistemic proxy measures.
- `relations`: typed relation candidates to other claim families.
- `geometry`: native reserved (nullable/maskable via data fields) gauge-invariant observables.
- `provenance`: source references and construction method.
- `training`: eligibility, target declarations, target availability, and loss-mask
  compatibility metadata.
- `quality`: extraction confidence, notes, and warnings.

## Identifier Examples

```text
axc:sha256:<hash>
claimfam:<slug>:<hash>
claimstate:<date-or-context>:<hash>
src:<kind>:<hash>
span:<hash>
rel:<hash>
ctx:<slug-or-hash>
```

## Validation Rules

- `format` must be `AXC`.
- `format_version` must be `0.1.0`.
- Canonical claim text must not be empty.
- `temporal.valid_as_of` is required.
- Source dates after `valid_as_of` are rejected unless marked `target_only`.
- Relation confidence, when present, must be in `[0, 1]`.
- Probability-like epistemic measures must be in `[0, 1]`.
- Independent redundancy accepts values greater than `1` but rejects negatives.
- Binary truth fields such as `truth`, `is_true`, or `correct` are forbidden.
- Raw geometry matrices are rejected because extra geometry fields are forbidden.
- Target-only or future-facing fields must not be rendered into predictor text or
  predictor-side AXT tensors.

## Target and Mask Policy

AXC may declare future-facing or optional targets, but missing targets are never
negative examples. AXT compilation must convert target availability into explicit
loss masks.

Reserved mask concepts include:

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

Relation, provenance, and context objectives require deterministic negative or
hard-negative sampling metadata in AXT/AXP manifests.

## Example Shape

```json
{
  "format": "AXC",
  "format_version": "0.1.0",
  "ids": {},
  "claim": {},
  "surface_forms": {},
  "temporal": {},
  "context": {},
  "epistemic_state": {},
  "relations": [],
  "geometry": {"enabled": false, "gauge_policy": "gauge_invariant_observables_only"},
  "provenance": {},
  "training": {},
  "quality": {}
}
```
