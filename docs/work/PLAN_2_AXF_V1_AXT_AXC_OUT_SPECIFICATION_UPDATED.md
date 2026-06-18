# Plan 2 / 6 — AXF v1 Contract, AXT & AXC-out Specification

**Status:** Ready after P1 completion
**Roadmap position:** P2 of the consolidated Axiom v1 roadmap
**Primary role:** Interface lock / semantic contract
**Audience:** Codex, Opus, Claude, GPT implementation agents
**Project identity:** Axiom is a structured-native LLM training framework

---

## 0. Purpose

P2 defines the stable structured interfaces that make the rest of Axiom possible.

P1 created the provider-capable corpus and substrate construction layer: provider ingress, cache/replay, cascades, merge/disagreement traces, source registry, synthetic views, epistemic proxies, negative pools, gold-reference hooks, AXP enrichment, fixture corpus configs, and optional PDF ingress.

P2 must now turn those outputs into formal contracts:

```text
AXF v1          = semantic data family
AXC v1          = canonical claim-state capsule stream
AXP v1          = dataset package with source/provider/control manifests
AXT v1          = compiled tensor bundle interface for structured-native models
AXC-out v1      = structured model emission format
```

P2 does **not** build the compiler, the model, the geometry module, the training loop, or evaluation. It specifies exactly what those later systems must consume and emit.

The central risk P2 prevents:

```text
AXC -> capsule text -> tokenizer -> ordinary causal LM
```

That is not the target. Axiom remains LLM-compatible through a text projection, but the primary model I/O is structured claim-field data.

---

## 1. Non-negotiable project constraints

Axiom is a **structured-native LLM training framework**.

It must preserve:

```text
structured input  -> full-complexity core -> structured output
```

Text remains mandatory as:

```text
- a projection head,
- a baseline channel,
- a comparability interface,
- a secondary language-modeling objective.
```

Text is **not** the native substrate.

P2 must preserve the following design decisions:

1. **AXC/AXT/AXC-out are primary.**
2. **Plain text is a projection, not the system boundary.**
3. **Provider outputs are data-construction provenance, not training/evaluation calls.**
4. **Provider identity/slant is lateral context.**
5. **Time and lateral context remain distinct axes.**
6. **No binary truth labels.**
7. **Geometry is optional, experimental, ablatable, and gauge-invariant.**
8. **Fairness is matched source content + temporal cutoffs + splits + params + compute/FLOPs.**
9. **Token parity applies only inside text-rendered arms and text projection.**
10. **AXC-out raw emissions must be stored separately from interpreted projections.**

---

## 2. Inputs from P1

P2 must inspect and formalize all machine-readable outputs produced by P1.

Expected P1 outputs include:

```text
source registry
provider registry
provider traces
provider cache/replay records
provider cascade traces
provider merge/disagreement traces
source documents
source spans
claim candidates
claim families
relation candidates
synthetic views
epistemic proxy records
negative pools
gold/evaluation-reference hooks
AXC streams
AXP packages
AXP enrichment reports
ML/software benchmark fixture configs
optional PDF reader metadata
```

P2 must not assume those fields are perfect. It must define:

```text
- which fields become AXF v1 fields,
- which fields become AXT tensor inputs,
- which fields become AXC-out targets,
- which fields remain metadata only,
- which fields are provider/debug traces,
- which fields are forbidden from predictor input,
- which fields require masks.
```

---

## 3. Deliverables

P2 must create or update the following specification files:

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
```

If the repository already has older v0.1 specs, do not delete them unless the repo convention supports replacement. Prefer:

```text
spec/archive/v0_1/...
spec/AXF_V1.md
```

or clear deprecation headers.

P2 must also create:

```text
docs/work/P2_HANDOFF_TO_P3.md
```

This handoff file is mandatory. P3 will implement against it.

---

## 4. AXF v1 semantic contract

Define AXF v1 as the semantic family.

AXF v1 must include:

```text
AXC = claim-state capsule stream
AXP = dataset package
AXT = tensor bundle
AXC-out = structured model emission
```

AXF v1 must define:

```text
format names
extensions
versioning rules
backward compatibility rules
canonical hashing rules
forbidden fields
required provenance
temporal leakage rules
target-only masking rules
provider provenance rules
geometry gauge policy
field registry relationship
```

Forbidden truth-style fields must remain banned in all active semantic formats:

```text
truth
is_true
correct
is_correct
factuality
ground_truth
label_truth
proven_true
```

If evaluation references require labels, use terms such as:

```text
evaluation_reference
gold_reference
human_verified_reference
reference_relation
reference_span
```

These must not imply metaphysical truth.

---

## 5. AXC v1 capsule schema

P2 must define the canonical AXC v1 record shape.

Required conceptual sections:

```text
format
format_version
ids
claim
surface_forms
temporal
lateral_context
provider_context
epistemic_state
relations
provenance
negative_pools
evaluation_references
synthetic_views
geometry
training
quality
lineage
```

### 5.1 Identity fields

AXC v1 must distinguish:

```text
capsule_id
claim_family_id
claim_state_id
context_id
source_ids
provider_trace_ids
relation_ids
negative_pool_ids
evaluation_reference_ids
```

### 5.2 Time fields

Time is the fact-core drift axis, not lateral context.

AXC v1 must define:

```text
valid_as_of
observed_at
constructed_at
source_publication_date
retrieved_at
temporal_cutoff_policy
target_only fields
future_target fields
temporal_split eligibility
```

### 5.3 Lateral context fields

Lateral context must include provider/source/community/method/context fibers without flattening time into them.

Fields should include:

```text
field
subfield
community
method_context
venue_context
language/register
provider_context
extractor_context
source_context
```

### 5.4 Provider trace references

P1 introduced provider traces and cascade/merge/disagreement metadata. AXC v1 must reference those traces cleanly.

Define:

```text
provider_id
provider_family
provider_mode: deterministic | openai_compatible | python_callable | human
provider_config_hash
prompt_template_id
prompt_template_version
request_hash
response_hash
cache_key
replay_key
cascade_stage
escalation_reason
merge_strategy
merge_confidence
disagreement_score
disagreement_set_ref
```

No API keys or secrets may appear in AXC/AXP/AXT.

### 5.5 Epistemic proxy fields

Each epistemic proxy must include at least:

```text
value
method
confidence
basis
source_fields_used
provider_trace_ref optional
```

Required proxy slots:

```text
ontology_compatibility / ontic_compatibility
evidential_anchoring
transformation_pressure
independent_redundancy
uncertainty
stability/status
```

Independent redundancy must remain distinct from popularity/frequency.

### 5.6 Negative pools

AXC v1 must define typed negative pools.

At minimum:

```text
near_topic_negative
near_claim_negative
same_context_unrelated
same_source_unrelated
contradiction_candidate
supersession_candidate
temporal_negative
provider_disagreement_negative
hard_relation_negative
provenance_negative
```

Each negative item needs:

```text
negative_id
type
target_ref or source_ref
sampling_method
sampling_basis
confidence / hardness score
provider_trace_ref optional
```

### 5.7 Evaluation references

Gold/evaluation hooks from P1 must be formalized without truth-label contamination.

Define:

```text
evaluation_reference_id
reference_type
reference_span_ref
reference_relation_type optional
reference_status optional
human_verified optional
verification_method optional
allowed_split
not_predictor_visible boolean
```

Evaluation references must not be rendered into predictor text.

---

## 6. AXP v1 package layout

AXP v1 must formalize package directories for P1/P2/P3/P6.

Recommended package layout:

```text
dataset.axp/
  axiom.json
  README.md

  data/
    capsules.axc
    sources.axsrc
    relations.axr
    contexts.axctx
    provider_traces.axprovtrace
    negative_pools.axneg
    evaluation_references.axref

  reports/
    construction_report.json
    provider_report.json
    cascade_report.json
    disagreement_report.json
    leakage_report.json
    axf_validation_report.json

  splits/
    train.json
    validation.json
    test.json
    temporal_holdout.json
    out_of_domain.json

  controls/
    context_shuffle_manifest.json
    provenance_shuffle_manifest.json
    popularity_frequency_control_manifest.json

  manifests/
    files.json
    hashes.json
    licenses.json
    providers.json
    source_registry.json

  spec/
    axf_version.json
    schema_snapshot.json
    field_registry_snapshot.json
    vocabulary_registry_snapshot.json
```

AXP v1 must define which files are required, optional, generated, and future-reserved.

---

## 7. AXT v1 tensor bundle contract

AXT is the compiled tensor interface consumed by the structured-native model.

P2 must specify AXT, not implement the compiler.

AXT v1 must include tensor groups for:

```text
ids
claim
temporal
lateral_context
provider_context
provenance
relations
relation_neighborhoods
epistemic_state
negative_samples
geometry_observables
text_projection
targets
availability_masks
loss_masks
split_masks
metadata
```

### 7.1 Required tensor groups

Define each tensor group with:

```text
name
dtype
shape convention
ragged representation strategy
padding strategy
mask strategy
source AXC fields
target AXC-out fields if applicable
whether predictor-visible
whether target-only
```

### 7.2 Ragged data strategy

P2 must define how to encode variable-length structures:

```text
source spans
provenance lists
relation lists
relation neighborhoods
negative pools
synthetic views
text tokens
```

Preferred strategy:

```text
values tensor + offsets tensor + mask tensor
```

or another explicit, stable strategy.

### 7.3 Text projection tensors

Text projection is mandatory for LLM comparability.

Define:

```text
text_input_ids
text_attention_mask
text_labels
text_loss_mask
text_render_mode
special_token_registry
```

Text projection tensors must be secondary to structured tensors.

### 7.4 Provider context tensors

P1 provider traces must map into AXT.

Define fields such as:

```text
provider_id_idx
provider_family_idx
provider_mode_idx
cascade_stage_idx
escalation_reason_idx
merge_confidence
disagreement_score
provider_vote_distribution optional
```

Provider identity/slant is lateral context.

---

## 8. AXC-out v1 structured output schema

AXC-out is the primary structured model emission format.

P2 must define AXC-out so P3/P4/P6 can implement against it.

AXC-out must include three output layers:

```text
raw_emission
validated_axc_out
interpreted_projection
```

### 8.1 Raw emission

Raw model outputs before schema repair or interpretation.

Must be stored for scoring and debugging.

May include logits, distributions, IDs, masks, and structured decoder emissions.

### 8.2 Validated AXC-out

Schema-validated structured output.

Must include:

```text
claim_state_prediction
relation_predictions
provenance_predictions
epistemic_predictions
temporal_predictions
geometry_observables
text_projection_ref
confidence/calibration fields
```

### 8.3 Interpreted projection

Human-readable or downstream-readable interpretation.

The interpreter may format, normalize, validate, or reject outputs.

It may not:

```text
add evidence not emitted by the model
repair relations using hidden graph oracle data
inject future information
turn uncertainty into truth labels
hide raw invalid emissions
score interpreter repairs as model competence
```

---

## 9. Target tensors and loss masks

P2 must define target availability and loss masking semantics.

Not every capsule has every target.

Required masks:

```text
loss_mask_text_projection
loss_mask_relation_prediction
loss_mask_provenance_recovery
loss_mask_stability_prediction
loss_mask_uncertainty_calibration
loss_mask_future_summary
loss_mask_temporal_prediction
loss_mask_geometry_observables
loss_mask_provider_context
loss_mask_negative_sampling
```

Required availability masks:

```text
available_relation_targets
available_provenance_targets
available_epistemic_targets
available_future_targets
available_geometry_targets
available_text_targets
available_evaluation_references
```

Missing target must not mean negative target.

P2 must explicitly distinguish:

```text
unknown
missing
not_applicable
not_predictor_visible
target_only
masked_by_split
available
```

---

## 10. Negative sampling metadata

P2 must define how P1 negative pools become AXT negative sample tensors.

Required metadata:

```text
negative_type
source_id / source_claim_family_id
target_id / target_claim_family_id
hardness_score
sampling_method
sampling_seed
provider_disagreement_ref optional
temporal_validity
split_eligibility
```

Negative samples must be reproducible and ablatable.

P2 must define how each negative type is eligible for which loss:

```text
relation contrastive loss
provenance recovery loss
context discrimination loss
temporal ordering loss
provider-disagreement auxiliary loss
```

No training losses are implemented in P2.

---

## 11. Field registry

Create a versioned field registry.

It must define every field used by AXC/AXP/AXT/AXC-out:

```text
field_name
semantic_owner
source format
AXC path
AXT tensor group
AXC-out target path
visibility: predictor | target_only | metadata | evaluation_only
requiredness
mask behavior
dtype if tensorized
allowed values / vocabulary
hash behavior
```

This registry is mandatory because P3 must compile tensors without guessing semantics.

---

## 12. Vocabulary registry

P2 must specify vocabulary snapshots for:

```text
claim types
relation types
status labels
provider ids
provider modes
cascade stages
escalation reasons
context fields
communities
method contexts
negative sample types
geometry observable names
special tokens for text projection
```

Special tokens are permitted only for text projection and text baselines. They are not the native Axiom substrate.

---

## 13. Geometry observable slots

Geometry is experimental and ablatable.

P2 must define reserved AXT/AXC-out geometry slots, including:

```text
geometry_enabled
context_node_ids
context_edge_ids
context_loop_ids
transport_path_ids
curvature_score
holonomy_norm
trace_summary
spectrum_summary
context_lability
geometry_loss_mask
geometry_ablation_mask
```

Raw connection matrices are not semantic AXF/AXC-out fields.

---

## 14. Text projection contract

P2 must define text projection as mandatory but secondary.

Required text projection modes:

```text
flat_text
structured_text
capsule_text
text_projection_from_structured_state
```

P2 must define:

```text
allowed special tokens
special token ID registry
rendering provenance
future-target exclusion rules
forbidden headings
text loss mask
projection-only fields
```

Text renderers must not expose:

```text
future_summary
target_timestamp
target_only
gold_reference
evaluation_reference
answer_key
ground_truth
```

---

## 15. Updated specs and docs

Update active documentation to point to the new contracts:

```text
README.md
CONCEPT.md
AGENTS.md
docs/concept/DATA_CONTRACT.md
docs/concept/TRAINING_BRIDGE.md
docs/concept/FALSIFICATION_HARNESS.md
docs/work/IMPLEMENTATION_ROADMAP.md
spec/AXF.md or spec/AXF_V1.md
spec/AXT_TENSOR_BUNDLE.md or spec/AXT_V1_TENSOR_BUNDLE.md
```

If documentation uses the old six-plan naming, preserve it if current. If it references the older ten-plan draft, update it to the accepted consolidated roadmap.

---

## 16. Tests and quality gates included in P2

Although validation is not a separate roadmap step, P2 must include tests.

Required test areas:

```text
tests/spec/test_field_registry_v1.py
tests/spec/test_axc_out_schema_v1.py
tests/spec/test_axt_tensor_contract_v1.py
tests/spec/test_loss_target_masks_v1.py
tests/spec/test_negative_sampling_spec_v1.py
tests/spec/test_provider_trace_spec_v1.py
tests/spec/test_no_truth_fields_v1.py
```

Tests should verify:

```text
AXC-out schema rejects truth labels
AXT spec includes target and loss masks
field registry maps required P1 outputs
provider trace fields are present and secret-safe
negative pool types are registered
text projection is marked secondary
geometry slots are gauge-invariant only
P3 handoff exists
```

Run:

```bash
ruff check .
ruff format .
mypy src
pytest
```

If the environment cannot run checks, record that in the P2 handoff.

---

## 17. Definition of done

P2 is complete only when:

```text
[ ] AXF v1 contract exists.
[ ] AXC v1 schema/spec exists.
[ ] AXP v1 package layout exists.
[ ] AXT v1 tensor bundle spec exists.
[ ] AXC-out v1 schema exists.
[ ] Field registry exists.
[ ] Vocabulary registry exists.
[ ] Loss and target mask semantics are defined.
[ ] Negative sampling metadata is defined.
[ ] Provider trace contract is defined.
[ ] Interpreter boundaries are defined.
[ ] Text projection contract is defined.
[ ] Geometry observable slots are defined.
[ ] P1 outputs are mapped to P2 interfaces.
[ ] No truth-label fields are introduced.
[ ] P2 handoff to P3 exists.
[ ] Tests/checks pass or failures are documented.
```

---

## 18. Out of scope

Do not implement:

```text
AXT compiler runtime
safetensors writer
runtime dataset
model architecture
learned geometry
training loop
evaluation run
benchmark scoring
```

Those belong to P3 and later.

---

## 19. Expected branch and commit

Suggested branch:

```text
plan/p2-axf-v1-axt-axc-out-contract
```

Suggested commit message:

```text
feat: define AXF v1 AXT and AXC-out contracts
```

---

## 20. P3 handoff requirements

The final handoff file must tell P3 exactly:

```text
where AXC/AXP inputs are located
which AXC fields compile into which AXT tensor groups
which field registries and vocab snapshots to load
which targets exist
which loss masks exist
which negative pools to compile
which provider traces to compile
which text projection tensors to create
which geometry slots to reserve
which AXC-out targets to emit later
which fields are metadata only
which fields must never enter predictor text
```

P3 must be able to implement the compiler from this handoff without inventing semantics.
