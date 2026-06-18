# AXF v1 Semantic Contract

AXF v1 is the semantic data family for Axiom, a structured-native LLM training
framework. It defines the stable interfaces used after P1 corpus ingress and
before P3 tensor compilation.

AXF v1 includes:

| Format | Extension | Role |
|---|---|---|
| AXC v1 | `.axc` | Canonical claim-state capsule stream |
| AXP v1 | `.axp/` | Dataset package with source/provider/control manifests |
| AXT v1 | `.axt`, `.axt.safetensors` | Compiled tensor bundle interface |
| AXC-out v1 | `.axc-out.json` | Structured model emission format |

Text remains mandatory as a projection, baseline, comparability interface, and
secondary loss surface. Text is not the native AXF boundary.

## Versioning

AXF v1 records must declare:

```text
format
format_version = "1.0.0"
schema_family = "AXF"
```

v0.1 artifacts may be read by compatibility adapters, but P3 must compile
against the v1 registry and handoff. v1 writers must not silently emit v0.1
records.

## Canonical Hashing

Canonical semantic records use deterministic UTF-8 JSON with sorted keys and no
volatile whitespace. Hashes are SHA-256 and written as either raw hex in legacy
contexts or `sha256:<hex>` in v1 registries and provider/cache records.

Hash scope must record whether the hash covers:

- raw source bytes;
- normalized source text;
- provider request payload;
- provider response payload;
- AXC semantic record;
- AXP package file;
- AXT manifest;
- AXC-out emission layer.

## Forbidden Truth-Style Fields

The following field names are forbidden in active AXF semantic records:

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

Evaluation material must use:

```text
evaluation_reference
gold_reference
human_verified_reference
reference_relation
reference_span
```

Those references are evaluation targets and review artifacts, not metaphysical
truth labels.

## Provenance And Provider Policy

Every AXF v1 artifact must preserve source provenance, construction lineage, and
hash traceability. Provider outputs from P1 are data-construction provenance.
They may become lateral context and compiler metadata, but providers are never
called in model training, scoring, benchmark judging, or evaluation.

No API key, secret token, raw credential, or provider account secret may appear
in AXC, AXP, AXT, AXC-out, provider traces, cache records, or reports.

## Temporal Policy

Time is distinct from lateral context. Predictor-visible sources must be at or
before `valid_as_of`. Future-facing fields may exist only as target-only or
evaluation-only material and must be masked out of predictor tensors and text
projection renderers.

Required temporal concepts:

- `valid_as_of`
- `observed_at`
- `constructed_at`
- `source_publication_date`
- `retrieved_at`
- `temporal_cutoff_policy`
- `target_only`
- `future_target`
- `temporal_split_eligibility`

## Geometry Policy

Geometry is experimental, learned in later plans, ablatable, and represented
only through gauge-invariant observables. AXF v1 reserves:

- `curvature_score`
- `transport_inconsistency`
- `holonomy_norm`
- `trace_summary`
- `spectrum_summary`
- `context_lability`

Raw connection matrices, basis-dependent gauge parameters, and geometry-only
success claims are not semantic AXF fields.

## Relationship To Registries

P3 must use:

- [FIELD_REGISTRY_V1.md](FIELD_REGISTRY_V1.md)
- [VOCABULARY_REGISTRY_V1.md](VOCABULARY_REGISTRY_V1.md)
- [LOSS_AND_TARGET_MASKS_V1.md](LOSS_AND_TARGET_MASKS_V1.md)
- [NEGATIVE_SAMPLING_V1.md](NEGATIVE_SAMPLING_V1.md)
- [PROVIDER_TRACE_V1.md](PROVIDER_TRACE_V1.md)
- [INTERPRETER_BOUNDARIES_V1.md](INTERPRETER_BOUNDARIES_V1.md)

Those files bind P1 corpus outputs to P3 AXT tensor groups and future AXC-out
targets.
