# AXP v0.1 Package Layout

AXP is the package directory format for complete AXF datasets.

Expected layout:

```text
<dataset>.axp/
  axiom.json
  README.md

  data/
    capsules.axc
    sources.axsrc
    relations.axr
    contexts.parquet
    provenance.parquet

  splits/
    train.json
    validation.json
    test.json
    temporal_holdout.json

  manifests/
    files.json
    hashes.json
    licenses.json
    construction_report.json
    leakage_report.json
    deduplication_report.json
    confound_controls.json

  compiled/
    README.md

  spec/
    axf_version.json
    schema_snapshot.json
```

The canonical v0.1 package helper writes `capsules.axc`, `sources.axsrc`, and
`relations.axr` as UTF-8 newline-delimited canonical JSON streams. Parquet
tables and compiled tensors are reserved for later stages.

## Root Manifest

The root manifest is `axiom.json`:

```json
{
  "format": "AXP",
  "format_version": "0.1.0",
  "package_id": "axp:dataset:version",
  "dataset_name": "Dataset",
  "created_at": "2026-01-01T00:00:00Z",
  "license": "unknown",
  "schema": {
    "axf": "0.1.0",
    "axc": "0.1.0",
    "axp": "0.1.0",
    "axt": "0.1.0"
  },
  "files": [],
  "counts": {},
  "temporal_policy": {},
  "construction": {},
  "hashes": {}
}
```

## Controls

AXP reserves control manifests for:

- temporal holdouts;
- context-shuffle controls;
- degree-preserving rewires;
- popularity/recency controls;
- embedding-only baselines;
- relation ablations;
- geometry-disabled ablations;
- provenance-disabled ablations.
