# AXT Runtime Data Interface

Status: Current | Updated: 2026-06-18 | Program: P3

AXT is Axiom's model-facing tensor bundle. It is the executable bridge from
AXC/AXP claim-field artifacts into structured tensors, targets, availability
masks, loss masks, and reversible metadata.

P3 implements this interface in `hcaps.axt`. It does not build the
structured-native model, train, evaluate, implement learned geometry math, or
decode AXC-out.

## Bundle Layout

The compiler writes a `.axt/` directory:

```text
bundle.axt/
  axiom.json
  tensors/*.safetensors
  registries/field_registry.json
  registries/vocabulary_registry.json
  manifests/compile_config.json
  manifests/tensor_manifest.json
  manifests/hashes.json
  manifests/source_index.json
  reports/validation_report.json
  reports/mask_report.json
  reports/missing_target_report.json
```

Numeric tensor groups are stored as safetensors. String IDs, source mappings,
rendered text audit material, vocabulary snapshots, and registry snapshots are
stored in sidecar JSON so P4 can map tensor indices back to AXC records without
parsing AXC directly.

## Tensor Groups

P3 writes these groups:

```text
ids
claim
temporal
lateral_context
provider_context
epistemic_state
provenance
relations
relation_neighborhoods
negative_samples
geometry_observables
text_projection
targets
availability_masks
loss_masks
split_masks
metadata
```

Ragged structures use `values + offsets + mask` conventions. Dense tensors use
record-major layout.

## Runtime Surface

Primary Python surfaces:

```python
from hcaps.axt import AxtCompileConfig, AxtDataset, compile_axt, validate_axt_bundle

compile_axt(AxtCompileConfig(...))
dataset = AxtDataset("bundle.axt")
record = dataset[0]
```

`AxtRecord` exposes structured groups:

```text
ids
claim
text_projection
temporal
lateral_context
provider_context
epistemic
provenance
relations
neighborhood
negative_samples
geometry_slots
targets
availability_masks
loss_masks
metadata
```

`AxtBatchCollator` preserves structured records. It does not collapse batches to
text-only token dictionaries.

## CLI

```bash
axiom axt compile --input examples/axf/v0_1/minimal_dataset.axp --output artifacts/axt/minimal.axt --config configs/axt/compile_smoke.yaml --allow-all-without-split
axiom axt inspect artifacts/axt/minimal.axt
axiom axt validate artifacts/axt/minimal.axt
axiom axt tensor artifacts/axt/minimal.axt --group relations
axiom axt compare artifacts/axt/a.axt artifacts/axt/b.axt
```

## Guardrails

- Text projection tensors are mandatory but secondary.
- Time tensors are separate from lateral context tensors.
- Provider identity and disagreement compile as provider/lateral context, not as hidden preprocessing.
- Missing targets produce availability/loss masks; they are not negative labels.
- P1 negative pools compile as typed negative samples.
- Geometry slots are gauge-invariant placeholders or observables only; raw connection matrices are not semantic AXT fields.
- Evaluation references remain metadata/evaluation-only and do not enter predictor tensors or predictor text.

## Next Consumers

P4 consumes AXT for the structured-native model stack.

P5 consumes AXT geometry slots and relation/context structure for learned
geometry and graph dynamics.

P6 consumes AXT targets, masks, provider context, text projection tensors, and
negative samples for training and experiment orchestration.
