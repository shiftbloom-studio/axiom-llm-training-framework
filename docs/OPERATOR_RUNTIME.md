# P6 Operator Runtime

The P6 operator surface provides the full training pipeline entrypoint plus local inspection and export commands for run artifacts.

Invocation `axiom full` starts the guided complete pipeline from data extraction through finished checkpoints. It prompts for source directory, temporal cutoff, training size, and extraction backend. Integrated extraction starts a managed local `llama-server` from the selected Hugging Face GGUF model as the primary construction provider, enables the built-in provider cascade, and uses Perplexity Sonar as remote escalation for low-confidence, warning-bearing, or high-impact construction records. The managed local server is shut down immediately after corpus construction; AXT compilation, training, scoring, and verdict generation run without provider services. Remote extraction uses Perplexity Sonar as the primary construction provider. API keys are read from or prompted into `PERPLEXITY_API_KEY` in-process and are not written to artifacts.

The subcommands are the post-training operator tools.

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

### Full-pipeline entry (interactive, one command)

```bash
axiom full
```

Follow prompts. Produces `runs/<generated-name>/` containing trained model checkpoints and the complete set of P6 artifacts. The command preserves provider-aware corpus ingress with cache/replay/merge traces, the complete P6 arm catalog, geometry path, masks, negatives, AXC-out, text projection, scoring, verdict, and reproducibility artifacts.

### Post-run operator commands

```bash
axiom run inspect runs/<run_id>
axiom run export runs/<run_id> --output artifacts/reproducibility_bundle.json
axiom run list runs/
```

The export bundle hashes reproducibility artifacts and records that no external LLM calls were used in training or evaluation.
