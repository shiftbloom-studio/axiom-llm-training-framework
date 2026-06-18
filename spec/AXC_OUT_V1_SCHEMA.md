# AXC-out v1 Structured Output Schema

AXC-out v1 is the primary structured model emission format. It is not a text
serialization. P2 specifies this schema so P7 can implement the decoder and
interpreter later.

Every AXC-out artifact must preserve four layers:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

## Raw Emission

`raw_emission` contains direct model outputs before schema repair or
interpretation. It must be stored for scoring and debugging.

Allowed content:

- logits;
- distributions;
- structured decoder ids;
- pointer emissions;
- mask emissions;
- calibration scalars;
- raw text-projection emissions.

Raw invalid emissions must remain visible.

## Validated AXC-out

`validated_axc_out` contains schema-validated structured output:

- `claim_state_prediction`
- `relation_predictions`
- `provenance_predictions`
- `epistemic_predictions`
- `temporal_predictions`
- `geometry_observables`
- `text_projection_ref`
- `confidence`
- `calibration`
- `validation_errors`

Validation errors must be recorded rather than hidden.

## Interpreted Projection

`interpreted_projection` is deterministic interpreter output for display or
downstream use. The interpreter may format, normalize, validate, or reject.

The interpreter may not:

- add evidence not emitted by the model;
- repair relations using hidden graph oracle data;
- inject future information;
- turn uncertainty into truth labels;
- hide raw invalid emissions;
- score interpreter repairs as model competence.

## Text Projection

`text_projection` is the secondary human-readable language surface. It must
record render mode, renderer version, source AXC-out layer, future-target
exclusion status, and text loss-mask provenance.

## Geometry

Allowed semantic geometry outputs are gauge-invariant observables:

- `curvature_score`
- `transport_inconsistency`
- `holonomy_norm`
- `trace_summary`
- `spectrum_summary`
- `context_lability`

Raw connection matrices are not AXC-out semantic fields.

## Forbidden Fields

AXC-out must not emit active truth-style fields. Evaluation references are
allowed only as references and must remain separate from model competence
scoring.
