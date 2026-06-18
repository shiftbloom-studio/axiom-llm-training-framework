# AXP v1 Package Layout

AXP v1 packages P1/P2/P3-ready AXF data. The package pins the source corpus,
provider provenance, controls, splits, registries, and validation reports used
by later tensor compilation.

## Recommended Layout

```text
dataset.axp/
  axiom.json
  README.md

  data/
    capsules.axc
    sources.axsrc
    relations.axr
    contexts.axctx
    provider_traces.axprovtrace
    negative_pools.axneg
    evaluation_references.axref

  reports/
    construction_report.json
    provider_report.json
    cascade_report.json
    disagreement_report.json
    leakage_report.json
    axf_validation_report.json

  splits/
    train.json
    validation.json
    test.json
    temporal_holdout.json
    out_of_domain.json

  controls/
    context_shuffle_manifest.json
    provenance_shuffle_manifest.json
    popularity_frequency_control_manifest.json

  manifests/
    files.json
    hashes.json
    licenses.json
    providers.json
    source_registry.json

  spec/
    axf_version.json
    schema_snapshot.json
    field_registry_snapshot.json
    vocabulary_registry_snapshot.json
```

## Required Files

Required for P3:

- `axiom.json`
- `README.md`
- `data/capsules.axc`
- `data/sources.axsrc`
- `data/relations.axr`
- `data/contexts.axctx`
- `splits/train.json`
- `splits/validation.json`
- `splits/temporal_holdout.json`
- `manifests/files.json`
- `manifests/hashes.json`
- `manifests/source_registry.json`
- `reports/construction_report.json`
- `reports/leakage_report.json`
- `spec/axf_version.json`
- `spec/schema_snapshot.json`
- `spec/field_registry_snapshot.json`
- `spec/vocabulary_registry_snapshot.json`

Required when provider-backed construction was used:

- `data/provider_traces.axprovtrace`
- `reports/provider_report.json`
- `reports/cascade_report.json`
- `reports/disagreement_report.json`
- `manifests/providers.json`

Required when negatives/evaluation references exist:

- `data/negative_pools.axneg`
- `data/evaluation_references.axref`

## Optional Or Future-Reserved Files

- `splits/test.json`
- `splits/out_of_domain.json`
- `controls/context_shuffle_manifest.json`
- `controls/provenance_shuffle_manifest.json`
- `controls/popularity_frequency_control_manifest.json`

## Root Manifest

`axiom.json` must record:

- package id and format version;
- AXF/AXC/AXP/AXT/AXC-out schema versions;
- file entries and hashes;
- source registry hash;
- provider registry hash;
- temporal cutoff policy;
- construction lineage;
- validation report references.

## Compatibility Note

Existing v0.1 helpers may store P1 reports under `manifests/`. P3 must accept
that layout through an adapter, but v1 packages should prefer `reports/` for
construction reports and `manifests/` for registries/hashes.
