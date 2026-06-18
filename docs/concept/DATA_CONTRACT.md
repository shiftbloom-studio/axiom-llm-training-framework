# Data Contract

The canonical record is `HoloCapsule` in `src/hcaps/schema/capsule.py`. It is a
strict Pydantic v2 model with `extra="forbid"` so unreviewed fields do not enter
the training substrate.

## Required Top-Level Fields

- `capsule_id`: stable `cap_` identifier.
- `schema_version`: semantic schema version.
- `created_at`, `updated_at`: timezone-aware timestamps.
- `claim`: normalized claim identity.
- `surface_forms`: text surfaces and source spans.
- `epistemic_state`: bounded uncertainty and epistemic scalars.
- `provenance`: at least one evidence/source record.
- `relations`: typed relation records.
- `context`: context fiber and temporal cutoff.
- `training_targets`: future training target placeholders.
- `quality`: extraction, license, and deduplication signals.
- `lineage`: pipeline and manifest ancestry.

## Identifier Rules

Identifiers must use explicit stable prefixes:

- `cap_` for capsules;
- `claim_` for claims;
- `src_` for provenance/source records;
- `doc_` for documents;
- `rel_` for relations;
- additional internal prefixes include `span_`, `ctx_`, `comm_`,
  `manifest_`, and `file_`.

## Temporal Rules

All timestamps must be timezone-aware. The `TemporalCutoff` and capsule-level
validators enforce:

- every model-visible source timestamp is `<= cutoff_at`;
- every future-facing prediction target timestamp is `> cutoff_at`;
- a `future_summary` target must be marked `future_facing` and must carry
  `target_timestamp`.

This catches obvious target leakage before tensorization or training.

## Epistemic Rules

`ontic_compatibility`, `evidential_anchoring`, `transformation_pressure`, and
`uncertainty` are bounded in `[0.0, 1.0]`.

`redundancy_effective_n` is an independent-community effective count. It is not
raw popularity, citation count, mention count, or view count. The schema names
the measure explicitly as `independent_community_effective_count`.

## Relation Rules

Relation types are enums. Unknown free strings are rejected. Initial relation
types are:

- `supports`
- `contradicts`
- `extends`
- `refines`
- `supersedes`
- `is_replicated_by`
- `uses_method_from`
- `shares_evidence_with`
- `same_claim_family_as`
- `near_but_distinct_from`
- `mentions`
- `related`

## Geometry Rules

HKR/geometric fields are optional and explicitly experimental. The schema allows
only gauge-invariant summaries:

- `loop_norm`
- `trace_summary`
- `spectrum_summary`
- `curvature_score`
- `context_lability`

Raw gauge matrices are not reported fields, and extra fields are forbidden.

## Storage Rules

JSONL is the canonical Step 1 format. Each line is a complete capsule and is
validated independently.

Parquet stores simple inspection columns and the canonical nested payload as
`record_json`. This makes the backend useful for simple flat scans while keeping
the full schema intact.

Manifests record file paths, roles, media types, sizes, record counts, hashes,
schema versions, and source document references.
