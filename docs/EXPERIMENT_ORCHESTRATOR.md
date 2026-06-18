# P6 Experiment Orchestrator

P6 adds a sequential local orchestrator for controlled Axiom experiment suites. It does not call external providers and does not perform LLM-as-judge evaluation.

## Arms

The runtime can express the required P6 arms:

```text
A_flat_text
B_structured_text
C_capsule_text
D_structured_native_no_geometry
E_structured_native_geometry
F_structured_native_no_provenance
G_structured_native_no_relations
H_structured_native_context_shuffle
I_structured_native_provider_shuffle
J_popularity_frequency_control
```

The default smoke suite runs:

```text
A, B, D, E, H, J
```

Text arms use text projection modes as comparison anchors. They are not treated as the native Axiom substrate.

`D_structured_native_no_geometry` uses the P5 non-geometric context mixer as a capacity-bearing control. It does not use learned transport, holonomy, curvature, or raw gauge matrices.

## Controls

Batch-time controls include:

- context shuffle;
- provider shuffle;
- provenance shuffle;
- no provenance;
- no relations;
- no context;
- popularity/frequency control;
- geometry off.

Targets and masks are preserved by controls. Future-only targets are not moved into predictor inputs.

## Fairness Report

Every suite writes:

```text
runs/<run_id>/comparisons/fairness_report.json
```

The report records source content, splits, extraction substrate, parameter counts, trainable parameters, steps, records seen, token counts for text/projection arms, and approximate FLOPs. Token parity is scoped only to text-rendered arms and text-projection metrics.

## CLI

```bash
axiom experiment plan configs/experiments/smoke_suite.yaml \
  --output runs/plans/smoke_plan.json

axiom experiment run configs/experiments/smoke_suite.yaml
axiom experiment score runs/<run_id>
axiom experiment compare runs/<run_id>
```

The smoke suite is intentionally tiny and proves the runtime path. It is not a research result.
