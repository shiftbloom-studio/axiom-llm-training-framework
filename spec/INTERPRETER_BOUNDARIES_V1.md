# AXC-out v1 Interpreter Boundaries

The interpreter converts structured AXC-out into deterministic display and
downstream projections. It is not a hidden evaluator and not an oracle repair
system.

## Required Layers

- `raw_emission`
- `validated_axc_out`
- `interpreted_projection`
- `text_projection`

Raw emissions must be stored and scored separately from validated and
interpreted outputs.

## Allowed Interpreter Actions

- validate schema conformance;
- normalize enum spellings by explicit registry;
- reject invalid structures;
- format human-readable summaries;
- produce text projection from emitted structure;
- attach validation errors and warnings.

## Forbidden Interpreter Actions

- add evidence not emitted by the model;
- repair relations using hidden graph oracle data;
- inject future information;
- turn uncertainty into truth-style labels;
- hide raw invalid emissions;
- score interpreter repairs as model competence;
- expose evaluation references in predictor text.

## Text Projection Guardrails

Text renderers must not expose:

```text
future_summary
target_timestamp
target_only
gold_reference
evaluation_reference
answer_key
ground_truth
```

Those terms are forbidden predictor exposure, not model inputs.
