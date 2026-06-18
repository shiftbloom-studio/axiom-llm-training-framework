# P6 Operator Runtime

The P6 operator surface provides local inspection and export commands for run artifacts.

## Artifact Layout

P6 runs follow this layout:

```text
runs/<run_id>/
  run_manifest.json
  environment.json
  git_state.json
  configs/
  arms/<arm_id>/
    train_log.jsonl
    validation_log.jsonl
    checkpoints/
    predictions/
    metrics.json
    arm_manifest.json
  comparisons/
    scores.json
    pairwise_metrics.json
    control_effects.json
    fairness_report.json
  verdict/
    verdict.json
    claims_ladder.json
    report.md
    research_card.md
  exports/
    reproducibility_bundle.json
```

Generated artifacts include hashes, config references, seeds, environment summaries, and code-state markers. They do not include secrets.

## CLI

```bash
axiom run inspect runs/<run_id>
axiom run export runs/<run_id> --output artifacts/reproducibility_bundle.json
axiom run list runs/
```

The export bundle hashes reproducibility artifacts and records that no external LLM calls were used in training or evaluation.
