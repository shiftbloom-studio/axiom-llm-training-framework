# P6 Handoff Final

Status: P6 implementation-complete for local smoke training, experiment orchestration, scoring, verdict reporting, and operator inspection/export.

P6 adds the first executable Axiom research runtime. It does not make benchmark claims and does not claim HKR proof.

## Implemented Runtime

Packages:

```text
src/hcaps/training/
src/hcaps/experiments/
src/hcaps/scoring/
src/hcaps/verdict/
src/hcaps/operator/
```

Config roots:

```text
configs/training/
configs/experiments/
configs/scoring/
configs/verdict/
```

Test roots:

```text
tests/training/
tests/experiments/
tests/scoring/
tests/verdict/
tests/operator/
```

## Smoke Command

```bash
axiom experiment run configs/experiments/smoke_suite.yaml
axiom verdict report runs/p6_smoke_suite --output runs/p6_smoke_suite/verdict/report.md
```

The smoke suite runs flat text, structured text, structured-native no-learned-geometry with a non-geometric context mixer, learned geometry, context-shuffle geometry, and popularity/frequency-control arms. It writes checkpoints, predictions, metrics, fairness reports, scores, verdicts, and reproducibility hashes.

## Preserved Guardrails

- Axiom remains a structured-native LLM training framework.
- AXC-out structured emission is scored separately from text projection.
- Text projection remains mandatory and secondary.
- No truth labels are introduced.
- Missing targets are masked, not converted into negative examples.
- No external LLM providers are used in training, scoring, judging, or evaluation.
- Geometry is ablatable and exported only through gauge-invariant observables.
- Fairness reports use source/content/splits/extraction/parameter/compute/schedule criteria.

## Next Action

Run the post-P6 project hardening and self-review pass.
