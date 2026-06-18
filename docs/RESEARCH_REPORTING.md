# P6 Research Reporting

P6 reports are research-runtime artifacts, not benchmark claims.

Reports must preserve these distinctions:

- structured AXC-out output is primary;
- text projection is mandatory but secondary;
- geometry is experimental, ablatable, and gauge-invariant;
- fairness is based on source content, temporal cutoffs, splits, extraction substrate, parameters, compute/FLOPs, and schedule;
- token parity applies only to text-rendered arms and text-projection metrics;
- external LLMs are not used in training, scoring, judging, or evaluation.

## Required Reports

P6 writes:

- arm metrics;
- pairwise comparisons;
- control effects;
- fairness report;
- verdict JSON;
- claims ladder JSON;
- Markdown verdict report;
- research card;
- reproducibility bundle.

## Interpretation

The smoke suite proves the runtime path:

```text
AXT -> training -> checkpoints -> predictions -> scores -> verdict
```

It does not establish that Axiom improves epistemic competence. A scaled or mini research run must be reported separately with its data, budgets, controls, and residual risks.
