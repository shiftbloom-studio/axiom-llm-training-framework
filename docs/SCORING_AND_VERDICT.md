# P6 Scoring And Verdict

P6 scoring is deterministic and local. It scores P6 artifacts without external LLM judges, remote scoring APIs, or provider calls.

## Scored Levels

The scorer distinguishes:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

Raw model emissions are preserved and scored separately from validated/interpreted outputs. The interpreter is not a hidden repair oracle.

## Metric Families

The scoring package reports visible component scores:

- `StructuredEpistemicScore`;
- relation metrics;
- provenance metrics;
- uncertainty/calibration metrics;
- temporal/leakage metrics;
- `GeometryEvidenceScore`;
- `TextProjectionScore`;
- fairness and audit compliance scores.

These are evidence metrics, not benchmark claims.

## Verdicts

The verdict system produces branch-level outcomes:

```text
proceed
proceed_with_caution
redesign
kill_branch
inconclusive
```

Branches include structured-native I/O, relation conditioning, provenance conditioning, geometry, epistemic router, text projection, and provider/context features.

The claims ladder is capped at Level 6. It explicitly forbids “HKR is proven.” Geometry success can be evidence for HKR-inspired epistemic dynamics, but P6 does not prove HKR.

## CLI

```bash
axiom score run runs/<run_id>
axiom verdict report runs/<run_id> --output runs/<run_id>/verdict/report.md
axiom verdict inspect runs/<run_id>/verdict/verdict.json
```
