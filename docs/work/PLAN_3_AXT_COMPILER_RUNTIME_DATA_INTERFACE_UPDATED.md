# Plan 3 / 6 — AXT Compiler & Runtime Data Interface

**Status:** Ready after P2 contract lock
**Roadmap position:** P3 of the consolidated Axiom v1 roadmap
**Primary role:** Executable compiler and runtime data interface
**Audience:** Codex, Opus, Claude, GPT implementation agents
**Project identity:** Axiom is a structured-native LLM training framework

---

## 0. Purpose

P3 implements the executable bridge from AXF/AXC/AXP into AXT tensor bundles and runtime datasets.

P2 defines the semantic contract.
P3 makes it real.

The output of P3 is not a model and not a training loop. It is a compiler/runtime layer that lets the later structured-native model consume full claim-field structure without falling back to capsule text as the primary substrate.

Central conversion:

```text
AXC / AXP
  -> field registry + vocabulary registry
  -> AXT tensor groups
  -> runtime dataset / batch interface
```

P3 must preserve Axiom's structured-native LLM identity:

```text
structured tensors are primary;
text projection tensors are secondary but mandatory;
AXC-out targets are compiled as structured targets;
loss and availability masks are first-class;
provider traces and disagreement are lateral context;
time remains separate from lateral context.
```

---

## 1. Precondition

Before starting P3, verify that P2 produced or updated:

```text
spec/AXF_V1.md
spec/AXC_V1_CAPSULE_SCHEMA.md
spec/AXP_V1_PACKAGE_LAYOUT.md
spec/AXT_V1_TENSOR_BUNDLE.md
spec/AXC_OUT_V1_SCHEMA.md
spec/FIELD_REGISTRY_V1.md
spec/VOCABULARY_REGISTRY_V1.md
spec/LOSS_AND_TARGET_MASKS_V1.md
spec/NEGATIVE_SAMPLING_V1.md
spec/PROVIDER_TRACE_V1.md
spec/INTERPRETER_BOUNDARIES_V1.md
docs/work/P2_HANDOFF_TO_P3.md
```

If P2 is missing, do not invent a parallel compiler. Stop and record the missing contract.

If P2 exists but has minor omissions, add small clarifying TODOs or minimal compatibility shims, but do not silently define new semantics in compiler code.

---

## 2. Deliverables

Create a new package:

```text
src/hcaps/axt/
  __init__.py
  config.py
  manifest.py
  registry.py
  vocab.py
  schema.py
  compiler.py
  writer.py
  reader.py
  dataset.py
  batch.py
  masks.py
  negatives.py
  neighborhoods.py
  text_projection.py
  provider.py
  geometry.py
  validation.py
  inspect.py
```

Add tests:

```text
tests/axt/
```

Add docs:

```text
docs/work/P3_AXT_COMPILER_HANDOFF_TO_P4.md
docs/concept/AXT_RUNTIME_DATA_INTERFACE.md
```

Update CLI:

```text
axiom axt compile
axiom axt inspect
axiom axt validate
axiom axt tensor
axiom axt compare
```

Update configs:

```text
configs/axt/compile_smoke.yaml
configs/axt/compile_ml_software_fixture.yaml
configs/axt/text_projection.yaml
configs/axt/no_geometry.yaml
configs/axt/provider_context_enabled.yaml
```

---

## 3. AXT compiler config

Implement a typed config model for compiling AXC/AXP into AXT.

Required fields:

```text
input_path
output_path
field_registry_path
vocabulary_registry_path
split_name optional
allow_all_without_split
render_text_projection_modes
max_text_length
include_provider_context
include_relation_neighborhoods
include_negative_samples
include_geometry_slots
include_evaluation_references
strict_temporal_masks
strict_schema_validation
hash_artifacts
seed
```

Config must be serializable to JSON/YAML and hashable.

---

## 4. AXT manifest

Implement an AXT manifest model.

Required fields:

```text
format = AXT
format_version
created_at
compiler_version
input_path
input_hash
source_format: AXC | AXP
field_registry_hash
vocabulary_registry_hash
config_hash
record_count
split_name
tensor_groups
artifacts
hashes
mask_summary
negative_sample_summary
provider_context_summary
text_projection_summary
geometry_slot_summary
warnings
```

The manifest must be written into every `.axt/` bundle.

Recommended AXT layout:

```text
bundle.axt/
  axiom.json
  tensors/
    ids.safetensors
    claim.safetensors
    temporal.safetensors
    lateral_context.safetensors
    provider_context.safetensors
    epistemic.safetensors
    provenance.safetensors
    relations.safetensors
    neighborhoods.safetensors
    negatives.safetensors
    geometry.safetensors
    text_projection.safetensors
    targets.safetensors
    masks.safetensors
  registries/
    field_registry.json
    vocabulary_registry.json
  manifests/
    compile_config.json
    tensor_manifest.json
    hashes.json
    source_index.json
  reports/
    validation_report.json
    mask_report.json
    missing_target_report.json
```

If `safetensors` is unavailable, P3 may implement a JSON/NPZ fallback only if clearly marked as non-final. Prefer safetensors as the target.

---

## 5. Field registry loader

Implement `registry.py` to load the P2 field registry.

Required functionality:

```text
load field registry
validate required fields
resolve AXC path -> tensor group
resolve AXC path -> AXC-out target path
resolve visibility: predictor | target_only | metadata | evaluation_only
resolve dtype
resolve mask behavior
resolve requiredness
```

The compiler must not hard-code semantic field mappings that belong in the registry.

Hard-coded fallback mappings are allowed only for smoke fixtures and must be clearly labelled.

---

## 6. Vocabulary registry compiler

Implement `vocab.py`.

Must compile stable integer vocabularies for:

```text
claim types
relation types
status labels
provider ids
provider families
provider modes
cascade stages
escalation reasons
context domains
communities
method contexts
source/media types
negative sample types
geometry observable names
special tokens for text projection
```

Vocabularies must be:

```text
stable
versioned
hashable
serializable
replayable
```

Unknown values must map to an explicit `<unk>` index, not crash silently unless strict mode is enabled.

---

## 7. AXC/AXP loader

Implement robust loading from:

```text
.axc stream
.axp package
```

AXP support must:

```text
locate data/capsules.axc
load source registry
load provider traces if available
load negative pools if available
load evaluation references if available
load splits
load reports/manifests
honor split selection
fail clearly if split is missing and allow_all_without_split=false
```

AXC support must:

```text
read canonical AXC stream
validate format/version
collect records
preserve original order unless deterministic sorting is configured
```

Do not use `.jsonl` as canonical AXF examples.

---

## 8. Tensor group compilation

Compile all AXT tensor groups defined by P2.

### 8.1 ID tensors

Compile:

```text
capsule_index
claim_family_index
claim_state_index
context_index
source_index
provider_trace_index
```

Also create lookup maps:

```text
index_to_capsule_id
index_to_claim_family_id
index_to_claim_state_id
index_to_context_id
index_to_source_id
index_to_provider_trace_id
```

### 8.2 Claim tensors

Compile:

```text
claim_type_idx
claim_family_idx
claim_state_idx
canonical_claim_text_ref
claim_text_hash
claim_length_features
```

Do not make claim text the primary substrate; text refs/projection tensors are secondary.

### 8.3 Temporal tensors

Compile time separately from lateral context.

Required:

```text
valid_as_of_timestamp
observed_at_timestamp
constructed_at_timestamp
source_publication_timestamp
retrieved_at_timestamp
time_delta_source_to_valid_as_of
time_delta_observed_to_valid_as_of
temporal_cutoff_mask
target_only_mask
future_target_mask
temporal_split_mask
```

Use stable numeric representation, e.g. Unix seconds or days since epoch. Document it.

### 8.4 Lateral context tensors

Compile:

```text
domain_idx
community_idx
method_context_idx
venue_context_idx
language_idx
register_idx
source_context_idx
```

Keep provider context separate unless the P2 registry says otherwise.

### 8.5 Provider context tensors

Consume P1 provider traces.

Compile:

```text
provider_id_idx
provider_family_idx
provider_mode_idx
provider_config_hash_ref
prompt_template_idx
cascade_stage_idx
escalation_reason_idx
merge_strategy_idx
merge_confidence
disagreement_score
provider_vote_distribution optional
cache_replay_flag
```

No secrets may enter AXT.

### 8.6 Epistemic tensors

Compile:

```text
ontology_compatibility_value
ontology_compatibility_confidence
evidential_anchoring_value
evidential_anchoring_confidence
transformation_pressure_value
transformation_pressure_confidence
independent_redundancy_value
independent_redundancy_confidence
uncertainty_value
uncertainty_confidence
status_idx
proxy_method_idx per proxy
proxy_basis_hash per proxy
```

Missing values require availability masks.

### 8.7 Provenance tensors

Compile ragged provenance:

```text
provenance_source_values
provenance_source_offsets
provenance_source_mask
source_date_values
source_license_idx
source_media_type_idx
source_hash_refs
span_refs optional
```

### 8.8 Relation tensors

Compile ragged relations:

```text
relation_type_values
relation_target_claim_family_values
relation_confidence_values
relation_offsets
relation_mask
relation_evidence_span_refs
```

### 8.9 Relation neighborhood tensors

Compile sampled neighborhoods.

Support:

```text
k-hop relation neighborhoods
relation type filters
source/target adjacency
edge features
node features
neighborhood masks
```

The n-ary/hypergraph path must remain open.

If only pairwise relations exist, implement pairwise now but reserve hyperedge fields:

```text
hyperedge_id
hyperedge_type
hyperedge_incidence_values
hyperedge_offsets
```

### 8.10 Negative sample tensors

Compile P1 negative pools into typed tensor groups.

Required:

```text
negative_type_idx
negative_source_idx
negative_target_idx
negative_hardness
negative_sampling_method_idx
negative_offsets
negative_mask
negative_loss_eligibility_mask
```

### 8.11 Geometry slot tensors

Reserve geometry slots.

Required even if geometry is off:

```text
geometry_enabled
context_node_ids
context_edge_ids
context_loop_ids
transport_path_ids
curvature_observable_placeholder
holonomy_norm_placeholder
trace_summary_placeholder
spectrum_summary_placeholder
geometry_loss_mask
geometry_ablation_mask
```

No raw connection matrices in semantic AXT.

### 8.12 Text projection tensors

Compile mandatory text projection tensors:

```text
flat_text_input_ids
flat_text_attention_mask
structured_text_input_ids
structured_text_attention_mask
capsule_text_input_ids
capsule_text_attention_mask
text_labels optional
text_loss_mask
special_token_ids
```

Use the tokenizer interface defined in the existing training bridge or extend it cleanly.

Text projection tensors must exclude future/evaluation-only target fields.

### 8.13 Target tensors

Compile target tensors for AXC-out training.

At minimum:

```text
relation_target_types
relation_target_ids
provenance_target_ids
status_target_idx
uncertainty_target
redundancy_target
future_summary_target_ref optional
geometry_observable_targets optional
text_projection_targets
```

Targets must only be active where availability masks allow them.

### 8.14 Loss and availability masks

Compile all masks defined in P2.

Required:

```text
available_relation_targets
available_provenance_targets
available_epistemic_targets
available_future_targets
available_geometry_targets
available_text_targets
loss_mask_relation_prediction
loss_mask_provenance_recovery
loss_mask_stability_prediction
loss_mask_uncertainty_calibration
loss_mask_future_summary
loss_mask_temporal_prediction
loss_mask_geometry_observables
loss_mask_text_projection
```

Missing target must not mean negative target.

---

## 9. Writer and reader

Implement writer/reader for AXT bundles.

Required:

```text
write bundle directory
write tensor groups
write registries
write manifest
write hashes
read bundle
inspect tensor group
validate hashes
load selected split
load selected tensor group lazily where possible
```

Do not require GPU.

Do not require torch to read metadata.

Torch conversion may be optional later, but P3's core reader should be framework-light.

---

## 10. Runtime dataset interface

Implement AXT runtime dataset classes.

Suggested classes:

```text
AxtBundle
AxtDataset
AxtRecord
AxtBatch
AxtBatchCollator
```

Dataset item must expose structured fields, not only text tokens.

Expected item fields:

```text
ids
claim
caption/text_projection optional
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

Batch collator must support:

```text
padding ragged tensors
stacking dense tensors
preserving masks
preserving ids
preserving text projection tensors
preserving negative samples
preserving relation neighborhoods
```

No model code in P3.

---

## 11. CLI

Add Typer commands under:

```text
axiom axt
```

Commands:

### 11.1 Compile

```bash
axiom axt compile \
  --input examples/axf/v0_1/minimal_dataset.axp \
  --output artifacts/axt/minimal.axt \
  --config configs/axt/compile_smoke.yaml
```

### 11.2 Inspect

```bash
axiom axt inspect artifacts/axt/minimal.axt
```

### 11.3 Validate

```bash
axiom axt validate artifacts/axt/minimal.axt
```

### 11.4 Tensor inspect

```bash
axiom axt tensor artifacts/axt/minimal.axt --group epistemic
```

### 11.5 Compare

```bash
axiom axt compare artifacts/axt/a.axt artifacts/axt/b.axt
```

CLI must produce clear errors for missing registries, missing splits, unsupported versions, and hash mismatches.

---

## 12. Configs

Create configs:

```text
configs/axt/compile_smoke.yaml
configs/axt/compile_ml_software_fixture.yaml
configs/axt/text_projection.yaml
configs/axt/no_geometry.yaml
configs/axt/provider_context_enabled.yaml
```

Each config must be minimal, readable, and executable against fixture data.

---

## 13. Tests and quality gates included in P3

Required tests:

```text
tests/axt/test_compile_minimal_axc.py
tests/axt/test_compile_minimal_axp.py
tests/axt/test_manifest_hashes.py
tests/axt/test_field_registry_mapping.py
tests/axt/test_vocabulary_registry.py
tests/axt/test_temporal_tensors.py
tests/axt/test_provider_context_tensors.py
tests/axt/test_epistemic_tensors.py
tests/axt/test_relation_tensors.py
tests/axt/test_neighborhood_tensors.py
tests/axt/test_negative_sample_tensors.py
tests/axt/test_geometry_slots.py
tests/axt/test_text_projection_tensors.py
tests/axt/test_target_and_loss_masks.py
tests/axt/test_axt_reader.py
tests/axt/test_axt_dataset.py
tests/axt/test_cli_axt.py
```

Tests must verify:

```text
AXC and AXP compile successfully
manifest hashes are stable
provider traces compile without secrets
negative pools compile into typed tensors
loss masks are not confused with negative labels
time is separate from lateral context
text projection is present but secondary
geometry slots are present and ablatable
runtime dataset returns structured records
CLI commands work on smoke fixtures
```

Run:

```bash
ruff check .
ruff format .
mypy src
pytest
```

If checks cannot run, document that in the P3 handoff.

---

## 14. Documentation updates

Create/update:

```text
docs/concept/AXT_RUNTIME_DATA_INTERFACE.md
docs/work/P3_AXT_COMPILER_HANDOFF_TO_P4.md
README.md
AGENTS.md
IMPLEMENTATION_ROADMAP.md
```

Docs must make clear:

```text
P3 does not build the model.
P3 does not train.
P3 does not evaluate.
P3 implements the structured-native tensor interface.
Text projection is compiled but secondary.
P4 consumes AXT.
P5 geometry consumes AXT geometry slots.
P6 training consumes AXT targets and masks.
```

---

## 15. Definition of done

P3 is complete only when:

```text
[ ] AXT compiler config exists.
[ ] AXT manifest exists.
[ ] Field registry is loaded and applied.
[ ] Vocabulary registry is loaded and applied.
[ ] AXC input compiles.
[ ] AXP input compiles.
[ ] Tensor groups are written.
[ ] Tensor groups are readable.
[ ] Runtime dataset returns structured records.
[ ] Runtime collator/batch preserves masks and ragged structures.
[ ] Provider context tensors compile.
[ ] Negative sample tensors compile.
[ ] Relation neighborhood tensors compile.
[ ] Geometry slots compile.
[ ] Text projection tensors compile.
[ ] Target and loss masks compile.
[ ] Manifests and hashes are written.
[ ] CLI commands exist and work on fixtures.
[ ] Tests/checks pass or failures are documented.
[ ] P3 handoff to P4 exists.
```

---

## 16. Out of scope

Do not implement:

```text
structured-native model architecture
learned geometry math
training loop
optimizer/scheduler
loss computation
AXC-out decoder
interpreter runtime
model evaluation
benchmark/verdict report
```

Those belong to later plans.

---

## 17. Expected branch and commit

Suggested branch:

```text
plan/p3-axt-compiler-runtime-interface
```

Suggested commit message:

```text
feat: compile AXF packages into AXT tensor bundles
```

---

## 18. P4 handoff requirements

The final handoff must tell P4/P5/P6:

```text
how to load AXT
which tensor groups are available
which masks are available
which fields are predictor-visible
which fields are targets only
which structured input groups the model must consume
which AXC-out target groups exist
which geometry slots exist
which provider context features exist
which negative samples exist
how relation neighborhoods are represented
how text projection tensors are represented
how to map tensor indices back to AXC IDs
```

P4 must be able to build the structured-native model stack without reading AXC directly.
