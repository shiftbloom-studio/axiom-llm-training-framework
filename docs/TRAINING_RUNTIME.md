# P6 Training Runtime

P6 adds the first executable training runtime for Axiom structured-native models.

Axiom remains a structured-native LLM training framework. The training boundary is AXT structured tensors into a P4/P5-compatible model, producing native AXC-out emissions plus a secondary text-projection head. Text projection is mandatory for LLM comparability, but it is not the primary substrate.

## What It Trains

`hcaps.training.AxiomTrainer` trains one experiment arm at a time:

```text
AXT bundle
  -> AxtDataset / AxtBatchCollator
  -> AxiomStructuredModel
  -> AXC-out raw emission heads
  -> masked multi-objective losses
  -> checkpoint, metrics, predictions, manifests
```

The trainer supports learned-geometry arms through the P5 `P4GeometryProvider` and capacity-bearing no-learned-geometry controls through `P4NonGeometricContextProvider`. Geometry regularizers are differentiable tensors, but only gauge-invariant observables are exported as semantic fields.

## Losses

The default registry includes:

- structured AXC-out claim-state/status loss;
- relation prediction loss with hard-negative metrics;
- provenance recovery loss;
- epistemic proxy loss;
- stability / temporal loss;
- uncertainty calibration loss;
- geometry observable and regularization loss;
- secondary text-projection loss.

Every component uses loss masks and target availability masks. Missing targets contribute zero loss and are never treated as negative examples.

## Checkpoints

P6 checkpoints include:

- model state;
- optimizer/scheduler state where configured;
- step and epoch;
- RNG state;
- training config and hash;
- AXT manifest hash;
- model config hash;
- loss registry version;
- curriculum state.

No provider secrets or external LLM state are stored.

## CLI

```bash
axiom train run configs/training/smoke_structured_native_no_geometry.yaml
axiom train resume runs/<run_id>/arms/<arm_id>/checkpoints/latest.pt
axiom train inspect runs/<run_id>
```

Standalone training configs expect an existing AXT bundle. The experiment smoke suite can compile the local minimal AXP fixture into AXT when its configured AXT path is missing.
