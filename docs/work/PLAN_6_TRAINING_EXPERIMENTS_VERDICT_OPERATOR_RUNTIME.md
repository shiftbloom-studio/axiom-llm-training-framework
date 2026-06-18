# PLAN 6 / 6 — Training, Experiments, Verdict & Operator Runtime

**Project:** Axiom — structured-native LLM training framework  
**Program:** P6 — Training, Experiments, Verdict & Operator Runtime  
**Status:** implementation plan, ready after P1–P5 are complete enough to provide AXT bundles, model stack, and geometry hooks  
**Language:** Python 3.14  
**Primary framework:** PyTorch  
**Geometry:** torch-native in-loop modules from P5; JAX remains reference/precompute only  
**Roadmap position:** P1 → P2 → P3 → P4 + P5 → **P6**  
**Commit target:** `feat: add training experiments verdict and operator runtime`

---

## 0. Executive Summary

P6 is the first plan where Axiom becomes an executable research system rather than a prepared substrate, interface, model stack, or geometry module.

P6 implements the runtime that can:

```text
AXP / AXT datasets
  -> structured-native model arms
  -> multi-objective training
  -> controlled experiment suites
  -> structured AXC-out scoring
  -> text-projection scoring
  -> geometry and non-geometry ablations
  -> verdict reports
  -> reproducible research artifacts
```

P6 does **not** merely train one model. It builds the system that can run the Axiom hypothesis honestly.

The central goal is to decide, on small but real runs, whether the structured-native Axiom path deserves scale-up, redesign, or branch-specific rejection.

P6 must keep these principles intact:

```text
1. Axiom remains a structured-native LLM training framework.
2. Structured AXC-out objectives are primary; text projection remains mandatory but secondary.
3. Learned geometry is tested as an ablatable inductive bias, not assumed true.
4. No external LLM or provider appears in training, scoring, judging, or evaluation.
5. Success is strong empirical evidence for HKR-inspired epistemic dynamics, not a claim of HKR proof.
6. Every positive result must survive controls that can kill boring explanations.
```

P6 is an all-or-nothing implementation program. It includes training, experiment orchestration, scoring, reports, operator CLI, tests, smoke fixtures, and reproducibility artifacts in one plan. These are not separate roadmap phases.

---

## 1. Governing Concept

### 1.1 Axiom is still an LLM training framework

P6 must not turn Axiom into a non-linguistic graph model.

Axiom remains LLM-compatible because every trained structured-native model must include:

```text
text projection head
tokenizer / text-projection vocabulary
secondary next-token or text projection loss
text-projection perplexity / token metrics
readable projection or interpreter surface
flat-text and structured-text comparison arms
```

But Axiom is not a standard token-in/token-out causal LM wrapper.

The primary training object is:

```text
AXT structured batch -> structured latent state -> AXC-out structured emission
```

Text is:

```text
projection
comparison interface
human-readable surface
secondary objective
baseline bridge
```

not the primary substrate.

### 1.2 Structured output is primary

The model must emit structured AXC-out targets through P4/P5 modules.

P6 trains and scores structured predictions such as:

```text
claim-state predictions
relation predictions
provenance recovery
stability / revision predictions
epistemic proxy predictions
uncertainty calibration
future-summary / temporal predictions where available
geometry observable predictions / conditioning diagnostics
```

The text projection must be trained and scored, but it must not become the only success criterion.

### 1.3 Geometry is the HKR test surface, not HKR proof

P6 must phrase geometry results as operational evidence.

Good claim:

```text
The learned geometry branch provides predictive value for epistemic claim-field dynamics
beyond structured non-geometric baselines under controls.
```

Bad claim:

```text
Axiom proves HKR.
```

Positive geometry results may be strong evidence for HKR-inspired epistemic dynamics, especially if they survive:

```text
geometry-off control
parameter-matched non-geometric baseline
context shuffle
provider shuffle
popularity/frequency control
relation ablation
temporal leakage audits
```

but P6 must not claim mathematical or physical proof.

### 1.4 Fairness is not token parity

The primary fairness rule is now:

```text
same source content
same temporal cutoff discipline
same splits
same extraction substrate
matched parameter budget
matched compute / FLOPs
matched training schedule where applicable
same available targets under masks
```

Token parity applies only within:

```text
flat_text vs structured_text text arms
text projection losses
text projection metrics
```

Do not compare structured-native and text-only arms by token count alone.

### 1.5 Training must not collapse into text serialization

P6 must reject any implementation that effectively becomes:

```text
AXC -> capsule text -> tokenizer -> normal CausalLM -> text output
```

Text-rendered arms are valid controls and baselines. They are not the target Axiom model.

---

## 2. Preconditions

P6 assumes P1–P5 have produced the following.

### 2.1 From P1 — substrate and provider ingress

P1 provides:

```text
source registry
provider traces
cache / replay records
cascade traces
merge / disagreement traces
synthetic views
epistemic proxies
negative pools
gold / evaluation-reference hooks
AXP report enrichment
ML/software benchmark fixture corpus
```

P6 uses these as:

```text
training/evaluation provenance
provider-context controls
negative sampling sources
evaluation-reference targets
input audit trail
```

P6 must not call those providers again.

### 2.2 From P2 — AXF v1 / AXT / AXC-out contract

P2 must define:

```text
AXF v1
AXC v1
AXP v1
AXT v1
AXC-out v1
field registry
vocabulary registry
structured target registry
target availability masks
loss masks
negative sampling metadata
interpreter boundaries
raw/validated/interpreted output levels
geometry observable registry
```

P6 must consume these contracts, not redefine them.

### 2.3 From P3 — AXT compiler and runtime interface

P3 provides:

```text
AXT bundles
runtime dataset classes
batch loader / collator
text projection tensors
structured feature tensors
target tensors
loss masks
availability masks
negative sample tensors
relation neighborhoods
split manifests
hash manifests
```

P6 must use AXT as the runtime data boundary.

### 2.4 From P4 — structured-native model stack

P4 provides:

```text
structured encoder
relation/provenance/context/provider conditioning
full-complexity core
epistemic router
structured decoder
AXC-out raw emission heads
text projection head
model config and ablation modes
geometry hook interface
```

P6 trains P4 modules under multiple arms and objectives.

### 2.5 From P5 — learned geometry and graph dynamics

P5 provides:

```text
claim-field graph runtime
context transition graph
learnable connection
parallel transport
holonomy / curvature-like observables
gauge-invariant summaries
geometry conditioning adapter
geometry-off and non-geometric controls
context/provider shuffle compatibility
```

P6 trains and compares geometry-on and geometry-off conditions.

---

## 3. Scope

### 3.1 In scope

P6 must implement:

```text
training runtime
optimizer / scheduler / seed control
checkpointing and resume
structured multi-objective loss system
loss-mask and target-availability handling
curriculum / complexity ramp
experiment arm orchestration
parameter/compute matching
run manifests and artifact hashing
scoring metrics
control suites
verdict logic
research report generation
operator CLI
configuration system
smoke and mini-run fixtures
quality gates and tests
```

### 3.2 Out of scope

P6 must not implement:

```text
new provider ingress features beyond tiny bug fixes
new AXF / AXT / AXC-out semantics except compatibility fixes
new model architecture beyond P4-defined interfaces
new geometry math beyond P5-defined interfaces
large-scale distributed training as the default path
human benchmark judging
automated LLM-as-judge scoring
external LLM calls in training/evaluation
claims of HKR proof
```

If P6 finds a missing contract in P2/P3/P4/P5, it must add a narrow compatibility issue / TODO and avoid silently inventing new semantics.

---

## 4. Package Layout

Add or extend these packages.

```text
src/hcaps/training/
  __init__.py
  config.py
  trainer.py
  state.py
  optim.py
  schedules.py
  curriculum.py
  losses.py
  loss_registry.py
  checkpointing.py
  seeds.py
  precision.py
  logging.py
  callbacks.py

src/hcaps/experiments/
  __init__.py
  arms.py
  configs.py
  orchestrator.py
  budgets.py
  matching.py
  controls.py
  suites.py
  artifacts.py
  comparison.py

src/hcaps/scoring/
  __init__.py
  structured.py
  text_projection.py
  calibration.py
  geometry.py
  provenance.py
  relations.py
  temporal.py
  aggregate.py

src/hcaps/verdict/
  __init__.py
  claims_ladder.py
  decision_rules.py
  report.py
  evidence_tables.py
  research_card.py

src/hcaps/operator/
  __init__.py
  inspect.py
  summaries.py
  export.py
```

Extend:

```text
src/hcaps/cli.py
configs/training/
configs/experiments/
configs/scoring/
configs/verdict/
docs/
tests/
```

If the repository already has equivalent package names, extend existing structure rather than duplicating.

---

## 5. Training Runtime

### 5.1 TrainingConfig

Implement a typed training configuration object.

Fields:

```text
run_name
seed
input_axt_path
output_dir
model_config_path
arm_name
train_split
validation_split
max_steps
max_epochs
global_batch_size
micro_batch_size
gradient_accumulation_steps
learning_rate
weight_decay
optimizer
scheduler
warmup_steps
cooldown_steps
clip_grad_norm
precision
checkpoint_interval
eval_interval
log_interval
save_optimizer_state
resume_from
structured_loss_weights
text_projection_loss_weight
geometry_loss_weight
curriculum_config
compute_budget_config
```

Rules:

```text
seed must be explicit
input AXT manifest hash must be recorded
all loss weights must be explicit
no default external provider usage
no benchmark claims from training logs
```

### 5.2 Trainer

Implement:

```text
AxiomTrainer
```

Responsibilities:

```text
load AXT dataset
instantiate P4/P5 model from config
run forward pass
compute structured and text losses
apply target/loss masks
backpropagate
clip gradients
step optimizer/scheduler
log metrics
save checkpoints
resume checkpoints
run validation hooks
write manifests
```

Minimum methods:

```python
class AxiomTrainer:
    def fit(self) -> TrainingRunResult: ...
    def train_step(self, batch) -> StepResult: ...
    def validation_step(self, batch) -> StepResult: ...
    def save_checkpoint(self, step: int) -> Path: ...
    def load_checkpoint(self, path: Path) -> None: ...
```

### 5.3 Torch dependency boundary

P6 is allowed to depend on PyTorch.

But previous layers must remain lightweight:

```text
schema
format
storage
provider ingress
AXF validation
falsification prep
```

Do not move torch into lower layers unnecessarily.

### 5.4 Precision modes

Support at least:

```text
fp32
bf16 if available
```

Optional:

```text
fp16
compile_mode if torch.compile is available
```

Do not require GPU for smoke tests.

### 5.5 Checkpoint contents

Each checkpoint must include:

```text
model state
optimizer state if configured
scheduler state
step
epoch
rng state
training config hash
AXT input manifest hash
model config hash
loss registry version
curriculum state
```

Do not checkpoint raw provider secrets.

---

## 6. Multi-Objective Loss System

### 6.1 Loss architecture

Implement a modular loss registry.

```text
LossRegistry
LossComponent
LossResult
LossMaskBundle
TargetAvailabilityBundle
```

Every loss component must declare:

```text
name
required_predictions
required_targets
required_masks
weight
normalization
missing-target behavior
metric outputs
```

### 6.2 Required loss components

Implement at least these loss components.

#### 6.2.1 Structured AXC-out reconstruction / prediction loss

Targets:

```text
claim-state fields
status / stability fields
structured output validity fields where applicable
```

Must use availability masks.

#### 6.2.2 Relation prediction loss

Targets:

```text
relation type
relation target candidates
support / contradiction / supersession / related classes
```

Must support negative samples from P1/P3.

Metrics:

```text
relation accuracy
macro F1
micro F1
AUPRC where applicable
hard-negative accuracy
```

#### 6.2.3 Provenance recovery loss

Targets:

```text
source id
span id
source rank
provenance confidence
```

Metrics:

```text
top-1 source recall
top-k source recall
span recovery recall
mean reciprocal rank
```

#### 6.2.4 Epistemic proxy loss

Targets:

```text
ontology / ontic compatibility
evidential anchoring
transformation pressure
independent redundancy
uncertainty
```

Use regression or distributional targets as defined by P2.

Do not train truth labels.

#### 6.2.5 Stability / revision / temporal loss

Targets:

```text
status trajectory
future stability if available
revision / supersession signals
temporal holdout targets
```

Must enforce target-only masks and temporal cutoff discipline.

#### 6.2.6 Uncertainty calibration loss

Support:

```text
Brier-style losses
NLL / calibration-aware losses where targets exist
confidence regularization
```

Do not interpret uncertainty as truth.

#### 6.2.7 Geometry auxiliary / regularization losses

Allowed geometry losses:

```text
transport consistency where defined
loop observable prediction where targets exist
geometry diagnostic alignment where defined
flatness regularization as optional and low-weight
```

Forbidden geometry losses:

```text
hard-coded redundancy-curvature law
loss that forces HKR to be true
loss that encodes popularity as curvature
loss that rewards raw citation degree
```

P6 may regularize geometry for stability, but must not bake the target thesis into the objective.

#### 6.2.8 Text projection loss

Text loss is mandatory but secondary.

Targets:

```text
text projection token ids
text-rendered baseline targets
```

Metrics:

```text
cross entropy
perplexity
token accuracy if useful
```

Text projection exists to preserve LLM comparability. It is not the primary success criterion.

### 6.3 Loss masks and availability masks

Every batch must carry masks for:

```text
loss_mask_relation
loss_mask_provenance
loss_mask_epistemic
loss_mask_stability
loss_mask_future_summary
loss_mask_geometry
loss_mask_text_projection
availability_relation
availability_provenance
availability_epistemic
availability_stability
availability_future_summary
availability_geometry
availability_text_projection
```

Rules:

```text
missing target != negative target
unavailable target contributes zero loss
loss denominators must account for masks
per-loss logs must show active target count
```

This is non-negotiable.

---

## 7. Curriculum / Complexity Ramp

P6 implements the training-time version of the spool-up principle.

### 7.1 CurriculumConfig

Fields:

```text
schedule_name
phase_boundaries
text_projection_weight_schedule
structured_loss_weight_schedule
relation_neighborhood_depth_schedule
side_channel_dropout_schedule
geometry_activation_schedule
context_dropout_schedule
negative_sample_hardness_schedule
```

### 7.2 Required curriculum modes

Implement at least:

```text
none
linear_complexity_ramp
staged_complexity_ramp
```

Example staged ramp:

```text
Phase 0: text projection + basic claim fields
Phase 1: add epistemic side channels
Phase 2: add relations and provenance
Phase 3: add relation neighborhoods and negative samples
Phase 4: add geometry conditioning
Phase 5: full AXC-out objective mix
```

### 7.3 Important guardrail

Curriculum must change exposure / weights, not data validity.

Never leak future-only targets into earlier predictor inputs.

---

## 8. Experiment Arms

P6 must train and compare controlled arms.

### 8.1 Minimum experiment arms

Implement these arms.

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

If compute is limited, support smoke subsets:

```text
A, B, D, E, H, J
```

But the full runtime must be able to express all arms.

### 8.2 ArmSpec

Define:

```text
arm_id
arm_name
source_content_id
input_axt_path
model_config
loss_config
geometry_mode
text_mode
control_transform
parameter_budget
compute_budget
training_schedule
seed
expected_outputs
```

### 8.3 Geometry arms

Required geometry comparison:

```text
E_geometry_on
D_geometry_off_parameter_matched
H_context_shuffle_geometry_on
```

The geometry-off model must be parameter-matched or capacity-matched as closely as possible.

Allowed strategies:

```text
dummy MLP with equal parameter count
non-geometric relation-conditioning module with matched width
frozen random geometry adapter only if clearly marked
```

### 8.4 Text arms

Text arms can use tokenized renderings, but they must not be presented as native Axiom.

They are comparison anchors:

```text
flat_text
structured_text
capsule_text
```

### 8.5 Controls

P6 must use controls from Step 4 / P3 / P5:

```text
context shuffle
provider shuffle
provenance shuffle
no provenance
no relations
no context
geometry off
relation ablation
popularity/frequency control
degree-preserving rewire if implemented
```

---

## 9. Budget Matching

### 9.1 ComputeBudgetConfig

Implement budget tracking for:

```text
parameters
trainable parameters
forward FLOPs estimate where possible
step count
samples seen
tokens seen for text arms
structured records seen
wall-clock time
hardware info
```

### 9.2 Fairness reports

Every experiment suite must output:

```text
parameter count per arm
trainable parameter count per arm
records seen per arm
source content id per arm
split id per arm
training steps per arm
estimated FLOPs per arm where available
text tokens seen for text/projection arms
structured target counts per loss
```

If perfect FLOP parity is not possible, the report must state the deviation.

### 9.3 Forbidden comparisons

Do not report a headline comparison if:

```text
source content differs unexpectedly
temporal cutoffs differ
splits differ
one arm used future target input
one arm had materially more trainable parameters without reporting
one arm used provider calls during evaluation
```

---

## 10. Scoring System

P6 implements scoring on both structured outputs and text projections.

### 10.1 Structured AXC-out scoring

Score raw and validated structured outputs separately.

Required output levels:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

Metrics must identify which level they score.

### 10.2 Relation metrics

Implement:

```text
relation_type_accuracy
relation_macro_f1
relation_micro_f1
support_contradiction_f1
hard_negative_accuracy
relation_target_recall_at_k
```

### 10.3 Provenance metrics

Implement:

```text
source_recall_at_1
source_recall_at_k
span_recall_at_k
provenance_mrr
provenance_calibration
```

### 10.4 Epistemic metrics

Implement:

```text
epistemic_proxy_rmse
epistemic_proxy_mae
stability_accuracy
revision_detection_f1
uncertainty_brier
expected_calibration_error
reliability_bins
```

### 10.5 Temporal metrics

Implement:

```text
temporal_holdout_loss
temporal_stability_prediction
outdated_belief_detection
future_target_mask_violation_count
temporal_leakage_count
```

### 10.6 Geometry metrics

Implement:

```text
geometry_on_minus_off_delta
context_shuffle_degradation
provider_shuffle_degradation
geometry_observable_stability
geometry_signal_vs_popularity_control
curvature_observable_correlation_with_revision
transport_consistency_score
```

These are evidence metrics, not proof metrics.

### 10.7 Text projection metrics

Implement:

```text
text_cross_entropy
text_perplexity
token_accuracy
text_length_normalized_loss
```

Text metrics should be reported as secondary for structured-native arms.

### 10.8 Aggregate metrics

Implement an aggregate score object, but avoid hiding details behind a single leaderboard number.

Recommended:

```text
StructuredEpistemicScore
TextProjectionScore
GeometryEvidenceScore
FairnessComplianceScore
AuditComplianceScore
```

Do not collapse everything into “AxiomScore” without component visibility.

---

## 11. Verdict System

P6 implements a decision system, not just metrics.

### 11.1 Hypotheses to evaluate

P6 should evaluate these hypotheses separately.

#### H1 — Structured data helps

```text
Structured text / capsule text beats flat text under text-arm fairness.
```

#### H2 — Structured-native I/O helps

```text
Structured-native no-geometry beats structured/capsule text under matched source/compute/params.
```

#### H3 — Geometry adds value

```text
Geometry-on beats parameter-matched geometry-off and degrades under context shuffle.
```

#### H4 — Provenance and relations matter

```text
No-provenance and no-relations ablations degrade relevant structured metrics.
```

#### H5 — Provider/lateral context carries signal

```text
Provider/context shuffles degrade relevant metrics beyond random noise.
```

#### H6 — Text projection remains viable

```text
Structured-native model maintains usable text projection performance.
```

### 11.2 Decision outcomes

Implement branch-level verdicts:

```text
proceed
proceed_with_caution
redesign
kill_branch
inconclusive
```

Apply these independently to:

```text
structured-native I/O
relation conditioning
provenance conditioning
geometry branch
epistemic router
text projection
provider-context features
```

Avoid one global yes/no unless evidence is strong.

### 11.3 Claims ladder

Implement a claims ladder:

```text
Level 0: Infrastructure works.
Level 1: AXC/AXT/AXC-out training is technically feasible.
Level 2: Structured arms beat flat text on at least one epistemic metric.
Level 3: Structured-native beats structured text under matched source/compute/params.
Level 4: Geometry adds predictive value beyond non-geometric structured baseline.
Level 5: Geometry effects survive context/provider shuffle and popularity controls.
Level 6: Results generalize across splits/domains and support HKR-inspired epistemic dynamics.
```

Forbidden:

```text
Level 7: HKR is proven.
```

P6 may say “strong empirical support” if evidence supports it. It may not say “proof.”

### 11.4 Thresholds

Default thresholds should be configurable, not hard-coded as universal scientific truth.

Example config:

```yaml
minimum_relative_improvement: 0.05
minimum_absolute_improvement: 0.01
max_fairness_parameter_delta: 0.02
max_compute_delta: 0.05
required_controls:
  - geometry_off
  - context_shuffle
  - popularity_frequency_control
  - temporal_leakage_audit
```

---

## 12. Experiment Orchestrator

### 12.1 ExperimentSuiteConfig

Fields:

```text
suite_name
input_axp
input_axt
output_dir
arms
seeds
budget_profile
trainer_config_template
model_config_template
scoring_config
verdict_config
```

### 12.2 Orchestrator

Implement:

```text
ExperimentOrchestrator
```

Responsibilities:

```text
expand arm specs
prepare per-arm configs
verify source/split compatibility
instantiate trainers
run arms
collect checkpoints/logs/metrics
score outputs
compute comparisons
generate verdict
write reports
```

Minimum methods:

```python
class ExperimentOrchestrator:
    def prepare(self) -> ExperimentPlan: ...
    def run(self) -> ExperimentRunResult: ...
    def score(self) -> ScoringResult: ...
    def verdict(self) -> VerdictReport: ...
```

### 12.3 Parallel / sequential execution

Support sequential execution first.

Optional:

```text
local multiprocessing per arm
SLURM launcher stub
GitHub Actions smoke only
```

Do not make distributed training a hard requirement for v1.

---

## 13. Artifact Layout

Every run writes an immutable artifact directory.

```text
runs/
  <run_id>/
    run_manifest.json
    environment.json
    git_state.json
    input/
      axt_manifest.json
      dataset_hashes.json
      splits.json
    configs/
      suite.yaml
      arms/
        <arm_id>.yaml
      training/
        <arm_id>.yaml
      model/
        <arm_id>.yaml
    arms/
      <arm_id>/
        train_log.jsonl
        validation_log.jsonl
        checkpoints/
        predictions/
          raw_emission.jsonl
          validated_axc_out.jsonl
          interpreted_projection.jsonl
          text_projection.jsonl
        metrics.json
        arm_manifest.json
    comparisons/
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

### 13.1 Manifest requirements

Every artifact must record:

```text
run_id
arm_id
input hashes
AXT manifest hash
model config hash
training config hash
loss config hash
code commit or dirty-state marker
Python version
dependency snapshot
hardware summary
seed
created_at
```

### 13.2 No secrets

Never write:

```text
API keys
provider credentials
private tokens
raw environment secrets
```

Provider traces from P1 may be referenced only by hashed IDs and safe metadata.

---

## 14. CLI

Extend `axiom` CLI.

### 14.1 Training commands

```bash
axiom train run configs/training/smoke_structured_native.yaml
axiom train resume runs/<run_id>/arms/<arm_id>/checkpoints/latest.pt
axiom train inspect runs/<run_id>
```

### 14.2 Experiment commands

```bash
axiom experiment plan configs/experiments/smoke_suite.yaml --output runs/plans/smoke_plan.json
axiom experiment run configs/experiments/smoke_suite.yaml
axiom experiment score runs/<run_id>
axiom experiment compare runs/<run_id>
```

### 14.3 Verdict commands

```bash
axiom verdict report runs/<run_id> --output runs/<run_id>/verdict/report.md
axiom verdict inspect runs/<run_id>/verdict/verdict.json
```

### 14.4 Operator commands

```bash
axiom run inspect runs/<run_id>
axiom run export runs/<run_id> --output artifacts/reproducibility_bundle.json
axiom run list runs/
```

CLI must fail clearly when:

```text
required AXT tensors are missing
loss masks are missing
arms are not parameter-matched enough
source content differs unexpectedly
temporal leakage audit fails in strict mode
```

---

## 15. Configs

Add configs.

```text
configs/training/
  smoke_flat_text.yaml
  smoke_structured_text.yaml
  smoke_structured_native_no_geometry.yaml
  smoke_structured_native_geometry.yaml
  mini_structured_native.yaml

configs/experiments/
  smoke_suite.yaml
  mini_axiom_suite.yaml
  geometry_controls_suite.yaml

configs/scoring/
  default_structured_scoring.yaml
  geometry_evidence_scoring.yaml
  text_projection_scoring.yaml

configs/verdict/
  default_claims_ladder.yaml
  conservative_thresholds.yaml
```

Smoke suite should run on CPU or a single modest GPU.

Mini suite may require GPU.

---

## 16. Documentation

Add or update:

```text
docs/TRAINING_RUNTIME.md
docs/EXPERIMENT_ORCHESTRATOR.md
docs/SCORING_AND_VERDICT.md
docs/OPERATOR_RUNTIME.md
docs/RESEARCH_REPORTING.md
docs/P6_HANDOFF_FINAL.md
```

Documentation must make clear:

```text
Axiom remains an LLM training framework.
Structured output is primary.
Text projection is mandatory but secondary.
Geometry success is evidence, not proof.
Fairness is source/content/compute/params based.
Token parity is only for text arms/projection.
External LLM providers are not used in training/evaluation.
```

---

## 17. Tests and Quality Gates

Tests are part of P6 implementation. They are not a separate roadmap step.

Add tests under:

```text
tests/training/
tests/experiments/
tests/scoring/
tests/verdict/
tests/operator/
```

### 17.1 Required test coverage

Implement tests for:

```text
loss masks prevent missing targets from contributing loss
unavailable target does not behave as negative target
trainer can run one step on smoke AXT
trainer checkpoint save/load roundtrip
trainer resume preserves step and RNG state
text projection loss runs as secondary loss
structured loss components aggregate correctly
geometry loss can be enabled/disabled
geometry-off model remains parameter matched within tolerance
experiment suite expands arms deterministically
context shuffle and provider shuffle controls are represented
fairness report detects parameter mismatch
scoring distinguishes raw_emission vs validated_axc_out vs text_projection
verdict rules produce proceed/redesign/kill/inconclusive outcomes
claims ladder never emits HKR proof language
CLI smoke commands execute on fixture configs
```

### 17.2 Quality commands

Run and fix:

```bash
ruff check .
ruff format .
mypy src
pytest
```

If GPU-specific tests exist, mark them separately and do not require them for default CI unless CI supports GPU.

---

## 18. Smoke Run Definition

P6 must include a smoke run that is small enough to execute quickly.

Minimum smoke run:

```text
input: fixture AXT bundle from P3
arms: flat_text, structured_native_no_geometry
steps: 2–5
batch size: tiny
model size: tiny
geometry: disabled
outputs: checkpoint, metrics, verdict report
```

Geometry smoke run:

```text
input: fixture AXT bundle with geometry slots
arms: structured_native_no_geometry, structured_native_geometry
steps: 2–5
outputs: geometry metrics, control report
```

The smoke run does not establish research claims. It proves the runtime path works.

---

## 19. Mini Research Run Definition

P6 should also support a small but meaningful mini-run.

Target:

```text
model size: ~10–50M parameters
corpus: P1 mini corpus / ML-software benchmark claims
arms: A, B, D, E, H, J at minimum
seeds: at least 1, preferably 3
compute: bounded and configurable
outputs: full verdict report
```

This mini-run may be executed after implementation. The implementation must support it.

Do not hard-code numbers as universal. Use config defaults.

---

## 20. Security and Safety

P6 must not use external LLM providers, human credentials, or network calls during training/evaluation.

Prohibited:

```text
OpenAI calls
Claude calls
LLM-as-judge calls
remote scoring APIs
secret-bearing environment dumps
provider cache refresh during scoring
```

Allowed:

```text
loading previously cached provider traces from P1
using hashed provider metadata as lateral context
training on pinned AXT bundles
local deterministic scoring
```

---

## 21. Definition of Done

P6 is complete when:

```text
1. A training runtime can train P4/P5-compatible models from AXT bundles.
2. Structured AXC-out losses and secondary text-projection losses are implemented.
3. Loss masks and target availability are honored everywhere.
4. Checkpointing and resume work.
5. Experiment suites can run multiple arms with matched budgets.
6. Geometry-on and geometry-off arms are expressible and comparable.
7. Context/provider/popularity controls are represented in experiment configs.
8. Structured, text, geometry, calibration, provenance, relation, and temporal metrics exist.
9. Verdict reports can produce proceed/redesign/kill/inconclusive outcomes per branch.
10. Claims ladder language avoids HKR proof overclaiming.
11. CLI commands exist for training, experiments, scoring, verdict, and run inspection.
12. Run artifacts include hashes, manifests, configs, metrics, and reports.
13. Smoke runs execute successfully.
14. Tests and quality gates pass.
15. Documentation explains how to run and interpret P6 outputs.
```

If P6 cannot produce a smoke run, it is not done.

If P6 can train but cannot score and issue a verdict report, it is not done.

If P6 reports performance but lacks fairness/control manifests, it is not done.

---

## 22. Final Output of P6

At the end of P6, the repository should contain:

```text
src/hcaps/training/
src/hcaps/experiments/
src/hcaps/scoring/
src/hcaps/verdict/
src/hcaps/operator/
configs/training/
configs/experiments/
configs/scoring/
configs/verdict/
docs/TRAINING_RUNTIME.md
docs/EXPERIMENT_ORCHESTRATOR.md
docs/SCORING_AND_VERDICT.md
docs/OPERATOR_RUNTIME.md
tests/training/
tests/experiments/
tests/scoring/
tests/verdict/
```

And it should be possible to run:

```bash
axiom experiment run configs/experiments/smoke_suite.yaml
axiom verdict report runs/<run_id> --output runs/<run_id>/verdict/report.md
```

without external services.

---

## 23. Agent Prompt

Use the following implementation prompt for Codex / Opus / GPT agent execution.

```text
You are implementing P6 for the Axiom repository.

Axiom is a structured-native LLM training framework. Its primary model I/O is structured Axiom data: AXT input tensors and AXC-out structured emissions. Text projection is mandatory for LLM compatibility and baseline comparison, but it is secondary.

Your task is to implement P6: Training, Experiments, Verdict & Operator Runtime.

Do not implement provider ingress, AXF contracts, AXT compiler, model architecture, or geometry math beyond compatibility fixes. Those belong to P1–P5.

Implement:
- training runtime
- optimizer/scheduler/checkpointing/resume
- structured multi-objective loss system
- target availability and loss masks
- text projection loss as secondary objective
- curriculum / complexity ramp
- experiment arm orchestration
- budget/fairness matching
- scoring metrics for AXC-out, provenance, relations, epistemic proxies, temporal behavior, geometry evidence, and text projection
- verdict system with claims ladder
- operator CLI
- run manifests and reproducibility artifacts
- smoke configs and tests

Hard rules:
- No external LLM calls in training or evaluation.
- No truth labels.
- No HKR proof claims.
- Geometry success is evidence, not proof.
- Missing target does not mean negative target.
- Text projection must remain present.
- Structured output must remain primary.
- Every headline comparison must include fairness/control manifests.

Run and fix:
ruff check .
ruff format .
mypy src
pytest

Expected commit message:
feat: add training experiments verdict and operator runtime
```
