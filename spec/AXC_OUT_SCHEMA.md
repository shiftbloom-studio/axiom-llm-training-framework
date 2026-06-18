# AXC-out v0.1 Structured Emission Schema

AXC-out is the structured Axiom model emission format.

This is a pre-runtime specification stub. P2 must finalize the contract and P7 must implement the structured decoder, deterministic interpreter, and text projection. This file does not implement AXC-out runtime behavior.

## Required Output Layers

Every AXC-out artifact must preserve:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

## Layer Meanings

| Layer | Meaning |
|---|---|
| `raw_emission` | Direct model output before schema validation or repair. Stored and scored directly. |
| `validated_axc_out` | Schema-validated structured output. Validation failures remain visible. |
| `interpreted_projection` | Deterministic interpreter output for display or downstream use. |
| `text_projection` | Language surface generated from the model state or interpreted structure. |

## Reserved Structured Fields

AXC-out should reserve fields for:

- claim-state prediction;
- relation prediction;
- provenance prediction;
- evidence-span pointers;
- epistemic values;
- stability and uncertainty outputs;
- future-summary or future-state predictions where target-available;
- temporal metadata;
- lateral-context metadata;
- gauge-invariant geometry observables;
- manifest/hash traceability.

## Guardrails

- No binary truth labels.
- No raw gauge matrices as canonical semantic fields.
- No hidden oracle repairs.
- No future information added by the interpreter.
- Missing targets require masks and must not become negative examples.
- Raw emissions must remain available for scoring and audit.

## Geometry

Allowed canonical observables include:

- curvature score;
- transport inconsistency;
- holonomy norm;
- normalized trace;
- spectrum summary;
- context-lability score.

Raw learned connection matrices may exist in experiment artifacts, but they are not AXC-out semantic fields.
