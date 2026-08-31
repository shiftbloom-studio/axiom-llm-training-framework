# PLAN 5 / 6 — Learned Geometry & Claim-Field Graph Dynamics

**Project:** Axiom — structured-native LLM training framework  
**Program:** P5 — Learned Geometry & Claim-Field Graph Dynamics  
**Status:** implementation plan, ready after P3 AXT compiler/runtime and P4 model-stack hooks  
**Language:** Python 3.14  
**Primary framework:** PyTorch  
**Reference framework:** JAX for off-loop math checks / precompute only  
**Roadmap position:** P1 → P2 → P3 → P4 + **P5** → P6  
**Commit target:** `feat: add learned claim-field geometry module`

---

## 0. Executive Summary

P5 implements Axiom's first real **learned claim-field geometry** workstream.

It turns the claim-field graph compiled by AXT into a differentiable geometry module that can be plugged into the P4 structured-native model stack and trained later by P6. It introduces learnable context-transport connections, path transport, loop / holonomy computation, curvature-like observables, gauge-invariant summaries, graph-dynamics features, and geometry-specific ablation modes.

P5 is not a metaphysical proof engine. It does not prove HKR, objectivity, truth, or physical drift. It implements a falsifiable inductive bias:

```text
If context transport structure matters,
then learned gauge-invariant geometry should improve structured epistemic modeling
beyond parameter-matched non-geometric baselines,
and its signal should degrade under context shuffle / graph controls.
```

P5 must preserve three hard truths:

```text
1. Geometry is experimental, not assumed true.
2. Geometry must be ablatable and parameter-matched against non-geometric controls.
3. Only gauge-invariant observables are stable public outputs.
```

The output of P5 is a set of torch-native modules and runtime objects that P4 can call and P6 can train. P5 does **not** implement the final training loop, run experiments, or report performance.

---

## 1. Governing Concept

### 1.1 Axiom's geometry is learned context transport

Axiom's geometry is not a static feature engineering layer. It is a trainable hypothesis about claim-field dynamics.

The model observes claim-state structure:

```text
claim families
relations
provenance
provider traces
lateral contexts
communities
methods
source domains
time-separated claim states
negative pools
```

and learns whether transitions among lateral contexts have stable transport structure.

A context transition might mean, for example:

```text
same claim moved from benchmark paper -> leaderboard report
same method claim moved from research paper -> GitHub README
same benchmark result moved from model card -> evaluation report
same claim rendered by deterministic extractor -> local provider -> remote provider
same claim carried by one community -> another community
```

The geometry module should learn whether transporting a claim-state representation around different context paths produces path-consistent or path-dependent results.

### 1.2 Time is not lateral context

P5 must preserve the conceptual distinction:

```text
Time = fact-core drift / evolution axis
Lateral context = field, provider, community, method, source, venue, framing, extractor
```

Do **not** flatten time into the context graph.

Time can condition transport and define temporal slices, but context loops must not be built by simply cycling through dates. Temporal transitions are not the same as lateral context transitions.

Valid examples:

```text
claim at t0 in context A -> claim at t0 in context B -> claim at t0 in context C -> context A
claim state trajectory t0 -> t1 used as temporal drift feature
claim at t0 with source dates <= cutoff only
```

Invalid example:

```text
context loop = 2020 -> 2021 -> 2022 -> 2020
```

Time can be part of the state. It is not itself the lateral gauge fiber.

### 1.3 Geometry is not popularity

The geometry module must not rediscover raw source count, mention count, degree, PageRank, or popularity and rename it curvature.

P5 must therefore expose the data and control hooks needed for P6 to compare:

```text
geometry_on
geometry_off_parameter_matched
context_shuffle
provider_shuffle
provenance_shuffle
degree_preserving_rewire
popularity_frequency_control
relation_ablation
```

P5 itself does not run the verdict, but it must make these controls technically possible.

### 1.4 Geometry is not truth

No geometry field may be named or treated as truth.

Good:

```text
curvature_score
transport_inconsistency
holonomy_norm
trace_normalized
spectrum_summary
context_lability
path_consistency
```

Bad:

```text
truth_score
objective_truth
proved_true
reality_score
correctness_curvature
```

P5 must remain compatible with AXF's no-truth-label rule.

---

## 2. Preconditions

P5 assumes these previous plans are available.

### 2.1 From P1

P1 provides enriched claim-field source material:

```text
provider traces
cache / replay records
cascade traces
merge / disagreement traces
source registry
claim candidates
claim families
relations
synthetic views
epistemic proxies
negative pools
gold-reference hooks
AXP report enrichment
```

P5 particularly needs:

```text
provider identity as lateral context
provider disagreement as possible geometry signal
relation candidates and neighborhoods
negative pools for future controls
source / community / method metadata
```

### 2.2 From P2

P2 must define the contracts:

```text
AXF v1
AXT tensor bundle schema
AXC-out schema
field registry
vocabulary registry
geometry observable registry
loss-mask and availability-mask rules
negative-sampling metadata
relation-neighborhood contract
interpreter guardrails
```

P5 must not invent public semantics that P2 did not define. If a needed geometry field is missing from P2, P5 must add a narrow compatibility extension and document it, not silently improvise.

### 2.3 From P3

P3 must compile AXT bundles containing enough graph/runtime structure for P5:

```text
claim node ids
claim family ids
claim state ids
context ids
provider ids
source ids
relation edges
relation types
relation neighborhoods
negative sample ids
lateral context features
temporal features
geometry slots
masks
split manifests
hashes
```

If P3 does not yet include a geometry tensor group, P5 must define the minimal runtime adapter, but must not rewrite the AXT compiler wholesale.

### 2.4 From P4

P4 must expose geometry hooks in the structured-native model stack:

```python
geometry_output = geometry_module(
    structured_state=structured_state,
    graph_batch=graph_batch,
    masks=masks,
)
```

P5 implements the real modules behind this hook.

---

## 3. P5 Objective

Implement a torch-native learned geometry module for Axiom claim-field graphs.

The implemented system must provide:

```text
1. Claim-field graph batch representation
2. Lateral context-transition graph construction
3. Relation / hyperedge graph support with n-ary path kept open
4. Learnable connection parameterization
5. Parallel transport along paths
6. Loop sampling and loop composition
7. Gauge-invariant holonomy / curvature observables
8. Geometry conditioning features for the P4 core
9. Geometry-specific regularizer terms for P6
10. Geometry-off and parameter-matched control modules
11. JAX reference / precompute path off the training critical path
12. Diagnostics, manifests, and exportable observables
```

P5 must be small enough to run on fixtures and smoke AXT bundles, but designed so the same interface can scale.

---

## 4. Non-Goals

P5 must **not** implement:

```text
full model architecture from scratch
training loop
optimizer / scheduler
benchmark evaluation
verdict report
claim extraction
provider ingress
AXT compiler rewrite
AXC-out interpreter
text projection head
external LLM calls
truth labels
physical HKR validation
```

P5 also must not hard-code a desired empirical law such as:

```text
curvature ∝ R^-1/2
```

If redundancy/curvature relationships are later measured, P6 must estimate them from held-out behavior. P5 may expose the observables needed for that test, but may not bake the result into the loss.

---

## 5. Package Layout

Create or extend:

```text
src/hcaps/geometry/
  __init__.py
  config.py
  graph.py
  batching.py
  context.py
  connection.py
  transport.py
  loops.py
  observables.py
  module.py
  controls.py
  regularizers.py
  diagnostics.py
  reference.py
  export.py
  cli.py
```

Optional if already existing conventions require different names:

```text
src/hcaps/model/geometry.py
```

But the preferred public module boundary is `hcaps.geometry`.

### 5.1 Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Geometry configuration models and enums |
| `graph.py` | Claim-field graph schemas and graph construction helpers |
| `batching.py` | Torch-ready geometry batch objects |
| `context.py` | Lateral context encoding and transition extraction |
| `connection.py` | Learnable connection parameterization |
| `transport.py` | Path transport and transition composition |
| `loops.py` | Loop/path sampling and masks |
| `observables.py` | Gauge-invariant geometry observables |
| `module.py` | Main torch `LearnedGeometryModule` and control modules |
| `controls.py` | Geometry-off, shuffled, rewire-compatible controls |
| `regularizers.py` | Loss-ready regularization terms for P6 |
| `diagnostics.py` | Debug summaries and health checks |
| `reference.py` | Optional off-loop JAX/numpy reference implementation |
| `export.py` | AXC-out compatible observable export |
| `cli.py` | Developer commands if the project CLI delegates subcommands |

---

## 6. Core Public API

P5 should expose a clear public API.

```python
from hcaps.geometry import (
    GeometryConfig,
    GeometryMode,
    ClaimFieldGraphBatch,
    LearnedConnection,
    LearnedGeometryModule,
    NoGeometryModule,
    PrecomputedGeometryModule,
    GeometryOutput,
)
```

### 6.1 Main forward contract

```python
class LearnedGeometryModule(nn.Module):
    def forward(
        self,
        *,
        node_states: torch.Tensor,
        graph_batch: ClaimFieldGraphBatch,
        context_features: torch.Tensor | None = None,
        temporal_features: torch.Tensor | None = None,
        masks: GeometryMasks | None = None,
    ) -> GeometryOutput: ...
```

Where:

```text
node_states:
  [num_nodes, hidden_dim]

graph_batch:
  relation edges, context transitions, loop indices, path indices, masks

context_features:
  lateral context features, not temporal drift alone

temporal_features:
  time/cutoff features used as conditioning, not as loop identity

GeometryOutput:
  conditioning_features
  observables
  regularizer_terms
  diagnostics
  exportable_axc_out_fields
```

---

## 7. Geometry Modes

Implement explicit modes.

```python
class GeometryMode(StrEnum):
    OFF = "off"
    PRECOMPUTED = "precomputed"
    LEARNED = "learned"
    REFERENCE = "reference"
```

### 7.1 OFF

No learned geometry. Returns zeros or learned parameter-matched placeholders depending on config.

Used for:

```text
geometry-off ablation
parameter-matched non-geometric baseline
smoke tests
```

### 7.2 PRECOMPUTED

Consumes geometry observables from AXT if present. Does not learn connection parameters.

Used for:

```text
offline analysis
JAX/numpy reference precompute
faster ablation
```

### 7.3 LEARNED

Main thesis-carrying mode. Learns context-transition connections end-to-end in torch.

Used for:

```text
P4 model integration
P6 training
geometry-on arms
```

### 7.4 REFERENCE

Off-loop diagnostic mode using numpy/JAX-style computations if available. This mode is not in the training graph.

Used for:

```text
small graph parity checks
mathematical debugging
operator sanity checks
```

---

## 8. Graph Representation

P5 must define a graph representation rich enough for claim-field dynamics but not so broad that P6 cannot use it.

### 8.1 Nodes

At minimum support:

```text
claim_state nodes
claim_family nodes
context nodes
provider nodes
source/provenance nodes
community nodes
method/domain nodes
```

Do not require every graph to contain every node type.

### 8.2 Edges

Support typed edges:

```text
claim_state -> claim_family
claim_state -> context
claim_state -> source
claim_state -> provider
claim_state -> relation target
context -> context transition
provider -> provider disagreement / merge relation
source -> community / domain
```

### 8.3 Hyperedges / n-ary path

The roadmap requires that the n-ary path remains open.

P5 should therefore represent hyperedges as incidence objects, even if the first implementation uses pairwise expansion for compute.

```python
class HyperedgeIncidence:
    hyperedge_id: Tensor
    node_id: Tensor
    role_id: Tensor
    hyperedge_type_id: Tensor
```

Examples of n-ary relations:

```text
claim supported by source under method in context
benchmark result comparing model A vs model B on dataset D with metric M
provider disagreement among N providers over same claim family
```

If full hypergraph message passing is not implemented in P5, P5 must still preserve the incidence metadata for later modules.

---

## 9. ClaimFieldGraphBatch

Define a torch-ready dataclass or Pydantic-free runtime object.

```python
@dataclass(frozen=True)
class ClaimFieldGraphBatch:
    node_ids: list[str]
    node_type_ids: torch.LongTensor
    node_to_claim_state: torch.LongTensor | None

    relation_edge_index: torch.LongTensor  # [2, num_edges]
    relation_type_ids: torch.LongTensor  # [num_edges]
    relation_confidence: torch.FloatTensor | None  # [num_edges]

    context_node_ids: list[str]
    context_edge_index: torch.LongTensor  # [2, num_context_edges]
    context_transition_type_ids: torch.LongTensor  # [num_context_edges]

    path_edge_index: torch.LongTensor  # [num_paths, max_path_len]
    path_mask: torch.BoolTensor  # [num_paths, max_path_len]

    loop_path_index: torch.LongTensor  # [num_loops, max_loop_len]
    loop_mask: torch.BoolTensor  # [num_loops, max_loop_len]

    hyperedge_incidence: HyperedgeIncidence | None
    temporal_features: torch.FloatTensor | None
    lateral_context_features: torch.FloatTensor | None
    provider_features: torch.FloatTensor | None
    masks: GeometryMasks
```

### 9.1 Stability requirements

- Must work on empty edge sets.
- Must work with one-node graphs.
- Must support CPU-only tests.
- Must not depend on PyTorch Geometric unless a fallback path exists.
- Must preserve stable ordering from AXT manifests.
- Must include masks for variable path/loop lengths.

---

## 10. Lateral Context Graph

### 10.1 Context sources

Construct lateral context from fields such as:

```text
field / domain
subfield
benchmark family
method context
source type
community
provider id
provider family
extraction mode
venue/source cluster
language/register
```

Time features may condition the geometry, but must not be the only basis for context transitions.

### 10.2 Context transition edges

Create context transition edges when at least one of these conditions holds:

```text
same claim family appears in multiple contexts
same source supports multiple context renderings
same relation appears across providers
same benchmark claim appears across model-card / paper / repo contexts
P1 merge/disagreement trace connects provider renderings
AXT relation neighborhood includes cross-context evidence
```

### 10.3 Context transition types

Define a registry:

```text
same_claim_cross_domain
same_claim_cross_provider
same_claim_cross_source_type
same_claim_cross_method
same_claim_cross_community
same_claim_cross_language
same_claim_cross_benchmark_family
provider_disagreement_transition
manual_reference_transition
unknown_transition
```

P5 should not hard-code project-specific strings deep in model code. Use registry ids.

---

## 11. Learnable Connection

### 11.1 Connection target

For each lateral context transition edge, learn a small connection matrix that transports a fiber representation:

```text
A_e ∈ so(k) or a stable low-rank/skew parameterization
T_e = exp(A_e) or a stable approximate transport operator
```

Where `k` is small:

```text
k = 8, 16, or 32
```

Do not use hidden_dim-sized full matrices by default. That is too expensive and too easy to overfit.

### 11.2 Skew-symmetric parameterization

Default:

```python
A = M - M.transpose(-1, -2)
```

or basis form:

```python
A_e = sum_i coeff[e, i] * B_i
```

where `B_i` are fixed skew-symmetric basis matrices.

Basis form is preferred for stability and parameter control.

### 11.3 Transport operator

Default transport:

```python
T = torch.linalg.matrix_exp(A)
```

For small `k`, matrix exponential is acceptable.

Optional alternative:

```text
Cayley transform
low-order exponential approximation
orthogonal projection / QR fallback
```

But if alternatives are implemented, they must be config-driven and tested.

### 11.4 Parameter-matched non-geometric baseline

P5 must include a parameter-matched module that consumes the same inputs and has similar parameter count but no path/loop geometry.

Example:

```text
NonGeometricContextMixer
```

It should combine context features through an MLP or attention-style mixer, not through holonomy/loop transport.

This is necessary for P6 to ask whether geometry itself matters beyond more parameters.

---

## 12. Parallel Transport

### 12.1 Path transport

Given path edges:

```text
e1, e2, ..., en
```

compose transport:

```text
T_path = T_en ... T_e2 T_e1
```

Implement batched path transport with masks.

### 12.2 Transported state

Given node fiber state:

```text
ψ_i ∈ R^k
```

transport along path:

```text
ψ_j_hat = T_path ψ_i
```

The module may use projected fiber states:

```python
fiber_state = fiber_projection(node_state)  # [num_nodes, k]
```

Do not assume that the main model hidden state itself is the geometry fiber.

### 12.3 Path consistency

If the graph has multiple paths between similar endpoints, compute path inconsistency:

```text
||T_path_a ψ - T_path_b ψ||
```

Expose this as a differentiable diagnostic/regularizer term.

---

## 13. Loop Sampling

### 13.1 Purpose

Loops allow the model to measure whether returning to a context through different paths produces a non-trivial holonomy.

This is the central geometry signal.

### 13.2 Loop types

Support at least:

```text
provider disagreement loops
same-claim cross-context loops
relation triangle loops
source-community-context loops
manual reference loops
synthetic view loops
```

### 13.3 Bounded loop sampling

Do not enumerate all loops.

Default loop sampler must be bounded:

```text
max_loops_per_batch
max_loop_length
max_loops_per_claim_family
max_context_degree
```

Loop selection should be deterministic under seed.

### 13.4 Empty loop handling

If no loops exist:

```text
return zero observables
set mask false
emit diagnostic no_loops=true
avoid NaNs
```

---

## 14. Holonomy and Curvature Observables

### 14.1 Holonomy

For a loop γ:

```text
H_γ = product_e∈γ T_e
```

Compute stable observables, not raw matrices.

### 14.2 Required gauge-invariant observables

Implement:

```text
holonomy_norm = ||H - I||_F
trace_normalized = Tr(H) / k
spectrum_abs_mean
spectrum_abs_max
loop_energy = mean squared transport inconsistency
curvature_score = normalized aggregate of holonomy_norm / loop length
path_consistency_score
context_lability_score
```

### 14.3 Export policy

Allowed in AXC-out / reports:

```text
curvature_score
holonomy_norm
trace_normalized
spectrum_summary
path_consistency_score
context_lability_score
loop_count
mode
```

Not allowed in AXC-out semantic fields:

```text
raw connection matrices
raw A_e
raw T_e
basis coefficients
unmasked provider secrets
source API payloads
```

Raw matrices may be saved only in debug artifacts when explicitly enabled, and must be marked non-semantic.

---

## 15. Gauge Discipline

P5 must be designed with gauge freedom in mind.

### 15.1 Public reports are gauge-invariant

Do not report individual matrix entries as meaningful.

### 15.2 Flat-as-possible prior

Curvature should not be free.

Add regularizers:

```text
connection_norm_regularizer
curvature_sparsity_regularizer
path_consistency_regularizer
context_smoothness_regularizer
```

These regularizers are passed to P6; P5 does not weight them globally.

### 15.3 No hard-coded target law

Do not encode a specific relationship between redundancy and curvature.

The model may observe redundancy features. It may not be forced to satisfy a preselected exponent.

P6 may later estimate:

```text
κ ~ R^-α
```

from held-out data.

---

## 16. GeometryOutput

Define a typed output object.

```python
@dataclass(frozen=True)
class GeometryOutput:
    mode: GeometryMode
    conditioning_features: torch.Tensor
    observables: GeometryObservables
    regularizer_terms: dict[str, torch.Tensor]
    diagnostics: GeometryDiagnostics
    axc_out_fields: dict[str, Any]
```

### 16.1 conditioning_features

Shape:

```text
[num_nodes, geometry_feature_dim]
```

These features feed the P4 full-complexity core / epistemic router.

### 16.2 observables

Torch tensors used for training and diagnostics.

### 16.3 regularizer_terms

Unweighted loss-ready values for P6.

Example:

```python
{
    "connection_norm": tensor,
    "loop_curvature": tensor,
    "path_consistency": tensor,
    "context_smoothness": tensor,
}
```

### 16.4 axc_out_fields

JSON-safe, gauge-invariant fields for structured emission.

---

## 17. Integration with P4 Model Stack

P5 must integrate with P4 through stable hooks.

### 17.1 Core integration

P4 should be able to do:

```python
geometry_output = self.geometry_module(
    node_states=structured_state.node_states,
    graph_batch=batch.graph,
    context_features=batch.context_features,
    temporal_features=batch.temporal_features,
    masks=batch.geometry_masks,
)

core_input = concat_or_fuse(
    structured_state.node_states,
    geometry_output.conditioning_features,
)
```

### 17.2 Router integration

Geometry observables may condition epistemic routing, but they must not dominate routing by default.

Provide config:

```text
router_use_geometry: bool
router_geometry_gate_init: low value
router_geometry_dropout
```

The router must work when geometry is off.

### 17.3 Decoder integration

AXC-out decoder receives geometry observables as candidate output fields, but P4/P6 decides which heads are trained.

P5 only provides:

```text
geometry head features
observable tensors
exportable fields
loss masks / availability masks passed through
```

---

## 18. Relation / Hypergraph Dynamics

P5 must not reduce all relationships to ordinary pairwise topic similarity.

### 18.1 Relation-aware transport

Relations can define paths and context bridges:

```text
supports
contradicts
supersedes
extends
mentions
related
uses_method_from
shares_evidence_with
```

### 18.2 Relation type conditioning

Connection parameters may depend on relation type embeddings:

```python
connection_coeffs = f(context_pair, relation_type, provider_context, temporal_condition)
```

But relation type must not become a truth label.

### 18.3 Hypergraph incidence path

Provide at least one function that converts hyperedge incidence into pairwise expansion for P5's first implementation, while preserving hyperedge ids and roles.

```python
def expand_hyperedges_to_pairwise(incidence: HyperedgeIncidence) -> PairwiseExpansion: ...
```

Later P5/P6 extensions may replace this with native hypergraph message passing.

---

## 19. Provider Disagreement Geometry

P1 introduced provider ingress, merge traces, and disagreement traces. P5 should make those first-class lateral context signals.

### 19.1 Provider nodes

Provider identities can be represented as nodes or context features.

```text
provider_id
provider_type
provider_family
local/remote/deterministic/python_callable
cascade_stage
```

Do not store API keys, secrets, or private payloads.

### 19.2 Disagreement edges

Provider disagreement traces can create edges/loops:

```text
provider_A rendering -> merged claim family -> provider_B rendering -> provider_A
```

These loops are valuable because extractor/provider slant is a concrete lateral context.

### 19.3 Merge confidence

Merge/disagreement signals may condition geometry, but must remain ablatable.

Config flags:

```text
use_provider_context
use_provider_disagreement_edges
provider_shuffle_control_ready
```

---

## 20. Controls and Ablations

Implement modules/control utilities required by P6.

### 20.1 Geometry off

```python
NoGeometryModule
```

Returns zeros and empty observables.

### 20.2 Parameter-matched non-geometric

```python
NonGeometricContextMixer
```

Consumes same context and graph features but does not compute transport, loops, or holonomy.

### 20.3 Context shuffle support

Provide a utility that can remap context ids in a `ClaimFieldGraphBatch` deterministically by seed.

P5 does not run the experiment but must make it possible.

### 20.4 Degree-preserving rewire compatibility

The graph representation must support swapped edge indices while preserving node degrees. P5 can provide utility helpers, but Step 4/P6 may own actual experiment generation.

### 20.5 Provider shuffle support

Provider-context shuffle should be possible separately from lateral context shuffle.

This is important because provider slant is a learned context axis but should not be silently confounded with domain context.

---

## 21. Regularizers and Loss-Ready Terms

P5 must return differentiable loss-ready terms. It does not decide final weights.

Required terms:

```text
connection_norm
curvature_energy
path_consistency
context_smoothness
transport_identity_bias
non_degenerate_usage
```

### 21.1 connection_norm

Penalize overly large connection parameters.

### 21.2 curvature_energy

Aggregate holonomy deviation from identity, mask-aware and loop-length normalized.

### 21.3 path_consistency

Compare endpoints transported by multiple paths where available.

### 21.4 context_smoothness

Encourage nearby contexts to avoid arbitrary high-frequency jumps unless data supports it.

### 21.5 transport_identity_bias

Initialize and mildly regularize transport toward identity so curvature is earned.

### 21.6 non_degenerate_usage

Prevent geometry features from collapsing to all zeros when geometry is on, but do not force high curvature.

This should be a weak diagnostic/regularizer, not a target outcome.

---

## 22. JAX / Reference Path

P5 must respect the framework split:

```text
PyTorch = in-loop learned geometry
JAX = off-loop reference / precompute only
```

### 22.1 Reference implementation

Add optional reference functions in `reference.py` using numpy or JAX if available.

These functions should compute, for small graphs:

```text
skew projection
matrix exponential / transport
path composition
loop holonomy
observables
```

### 22.2 No hard dependency if avoidable

If JAX is not already a dependency, make reference mode optional. Tests may skip if JAX is unavailable.

### 22.3 Parity checks

Provide small parity tests:

```text
torch and reference observables agree within tolerance on deterministic small graph
```

But do not make GPU JAX required for CI.

---

## 23. Diagnostics

Add geometry diagnostics that help later agents debug training.

Required diagnostics:

```text
num_nodes
num_edges
num_context_edges
num_paths
num_loops
mean_loop_length
mean_holonomy_norm
max_holonomy_norm
mean_trace_normalized
curvature_nonzero_fraction
connection_norm_mean
connection_norm_max
no_loops
empty_graph
geometry_mode
```

Diagnostics must be JSON-safe when exported.

---

## 24. Export to AXC-out

P5 should provide a function:

```python
def geometry_output_to_axc_out_fields(output: GeometryOutput) -> dict[str, Any]: ...
```

Allowed export shape:

```json
{
  "geometry": {
    "enabled": true,
    "mode": "learned",
    "gauge_policy": "gauge_invariant_observables_only",
    "curvature_score": 0.13,
    "holonomy_norm": 0.22,
    "trace_normalized": 0.91,
    "spectrum_summary": {
      "abs_mean": 1.0,
      "abs_max": 1.02
    },
    "path_consistency_score": 0.84,
    "context_lability_score": 0.18,
    "loop_count": 12
  }
}
```

Do not export raw connection matrices.

---

## 25. Config Files

Add:

```text
configs/geometry/geometry_off.yaml
configs/geometry/geometry_learned_smoke.yaml
configs/geometry/geometry_precomputed_smoke.yaml
configs/geometry/geometry_reference_smoke.yaml
```

Example fields:

```yaml
mode: learned
fiber_dim: 16
connection_rank: null
transport_operator: matrix_exp
max_loop_length: 4
max_loops_per_batch: 128
max_loops_per_claim_family: 8
use_provider_context: true
use_provider_disagreement_edges: true
use_relation_type_conditioning: true
regularizers:
  connection_norm: 0.0
  curvature_energy: 0.0
  path_consistency: 0.0
  context_smoothness: 0.0
export:
  include_raw_matrices: false
```

Regularizer weights are zero by default in config examples unless a smoke test explicitly needs nonzero values. P6 decides final training weights.

---

## 26. Documentation

Add:

```text
docs/GEOMETRY_MODULE.md
```

It must explain:

```text
what geometry means in Axiom
what it does not mean
why it is experimental
why only gauge-invariant observables are exported
how time differs from lateral context
how provider disagreement becomes lateral context
how geometry integrates with P4
how P6 will ablate it
why raw matrices are not semantic outputs
how JAX reference mode relates to torch-native training
```

Update:

```text
README.md
CONCEPT.md if necessary
IMPLEMENTATION_ROADMAP.md if necessary
AGENTS.md if necessary
```

Do not rewrite canonical docs unnecessarily. Patch only if P5 introduces new accepted terminology.

---

## 27. CLI and Developer Surface

If the project CLI is structured for this, add:

```bash
axiom geometry inspect-axt <bundle.axt>
axiom geometry sample-loops <bundle.axt> --output loops.json
axiom geometry compute-reference <bundle.axt> --output observables.json
axiom geometry smoke <bundle.axt> --config configs/geometry/geometry_learned_smoke.yaml
```

CLI should be developer-facing, not product-facing.

If full CLI integration would create too much churn, add scriptable functions and document them. Do not block P5 on CLI polish.

---

## 28. Tests and Quality Gates

Tests are part of P5 implementation, not a separate roadmap step.

Add:

```text
tests/geometry/
  test_graph_batch.py
  test_connection.py
  test_transport.py
  test_loops.py
  test_observables.py
  test_module_forward.py
  test_ablation_modes.py
  test_provider_context_geometry.py
  test_export_policy.py
  test_reference_parity.py
```

### 28.1 Required tests

#### Graph batch

```text
empty graph works
single-node graph works
graph with relation edges works
graph with context transitions works
hyperedge incidence expands without losing ids
stable ordering is preserved
```

#### Connection

```text
connection matrices are skew-symmetric
transport matrices have expected shape
identity initialization produces near-zero curvature
matrix_exp path has finite gradients
```

#### Transport

```text
path transport composes in correct order
masks ignore padded edges
empty path returns identity
```

#### Loops

```text
loop sampler is deterministic by seed
loop count respects bounds
no loops returns masked empty result
```

#### Observables

```text
identity loop has near-zero holonomy_norm
trace_normalized has expected range for identity
observables are finite
no raw matrices in export
```

#### Module forward

```text
learned module returns conditioning_features
learned module returns regularizer terms
learned module is differentiable
geometry_off returns correct zeros
non-geometric baseline is shape-compatible
```

#### Provider context

```text
provider context features can create transitions
disagreement edges can be included/excluded
provider shuffle utility is deterministic
```

#### Export policy

```text
AXC-out fields are JSON-safe
raw matrices are not exported by default
forbidden truth fields are absent
```

#### Reference parity

```text
torch observables match reference implementation on tiny deterministic graph
skip cleanly if optional reference dependency unavailable
```

### 28.2 Quality commands

Run:

```bash
ruff check .
ruff format .
mypy src
pytest
```

If optional JAX reference tests are skipped, the skip reason must be explicit.

---

## 29. Performance and Complexity Constraints

P5 must be serious but bounded.

### 29.1 Default sizes

```text
fiber_dim: 16
max_loop_length: 4
max_loops_per_batch: 128
max_context_degree: configurable
```

### 29.2 Avoid all-loop enumeration

Never enumerate all cycles in large graphs by default.

### 29.3 Vectorize where possible

Use batched torch operations for:

```text
skew projection
matrix exponential
path composition
observable computation
```

### 29.4 Keep CPU smoke viable

All fixture tests must run on CPU.

### 29.5 Do not prematurely optimize

CUDA/Triton/Rust is out of scope unless profiling later proves bottleneck.

---

## 30. Definition of Done

P5 is done when:

```text
[ ] hcaps.geometry package exists
[ ] ClaimFieldGraphBatch exists and handles empty/small graphs
[ ] lateral context transitions are represented separately from time
[ ] provider context/disagreement can be represented
[ ] learnable connection exists and is differentiable
[ ] path transport works with masks
[ ] loop sampler works deterministically
[ ] holonomy/curvature observables are computed
[ ] only gauge-invariant observables are exported
[ ] geometry-off module exists
[ ] parameter-matched non-geometric module exists
[ ] geometry output plugs into P4 geometry hook
[ ] regularizer terms are returned unweighted for P6
[ ] AXC-out geometry field export exists
[ ] optional reference path exists or is clearly deferred with interface stub
[ ] tests cover graph, connection, transport, loops, observables, ablations, export
[ ] docs/GEOMETRY_MODULE.md exists
[ ] configs/geometry smoke configs exist
[ ] ruff, mypy, pytest pass
```

P5 is **not** done merely because geometry feature fields exist in AXT. It is done only when there is a real torch-native learned geometry module with differentiable transport and gauge-invariant observables.

---

## 31. P5 Handoff to P6

Create:

```text
docs/work/P5_HANDOFF_TO_P6.md
```

or the repository's equivalent work-doc path.

It must include:

```text
1. Geometry module public API
2. Geometry config defaults
3. Graph batch schema
4. Required tensors from AXT
5. GeometryOutput fields
6. AXC-out geometry export fields
7. Regularizer terms and meanings
8. Ablation/control modes
9. Reference mode status
10. Known limitations
11. Recommended P6 training arms
```

### 31.1 Recommended P6 geometry arms

P5 should recommend, but not implement, these arms:

```text
structured_native_no_geometry
structured_native_non_geometric_context_mixer
structured_native_learned_geometry
structured_native_learned_geometry_context_shuffle
structured_native_learned_geometry_provider_shuffle
structured_native_learned_geometry_degree_rewire
structured_native_learned_geometry_popularity_control
```

---

## 32. Suggested Implementation Order Inside P5

Implement in this order:

```text
1. Configs and dataclasses
2. Graph batch representation
3. Lateral context transition extraction
4. Connection parameterization
5. Transport composition
6. Loop sampler
7. Observables
8. GeometryOutput and export policy
9. LearnedGeometryModule
10. NoGeometryModule and NonGeometricContextMixer
11. Regularizers
12. Provider-disagreement integration
13. Reference implementation / parity checks
14. Docs, configs, CLI, tests
15. P5 handoff to P6
```

Do not start with math experiments in notebooks. Start with stable interfaces and then fill the math.

---

## 33. Expected Pull Request

### Branch

```text
feature/p5-learned-geometry
```

### Commit message

```text
feat: add learned claim-field geometry module
```

### PR title

```text
P5: Learned Geometry & Claim-Field Graph Dynamics
```

### PR body must include

```text
- Summary
- Geometry modes implemented
- Public API
- AXT/P4 integration points
- AXC-out observable fields
- Ablation/control support
- Reference mode status
- Tests added
- What P5 deliberately does not do
- P6 handoff notes
```

---

## 34. Agent Instructions

You are implementing P5 only.

Do not reduce the plan into a simple graph feature extractor.

Do not implement a stock graph neural network and call it geometry.

Do not export raw matrices as semantic outputs.

Do not flatten time into context.

Do not use geometry to claim truth.

Do not run model training or benchmark evaluation.

Do not hard-code the expected empirical success of geometry.

Do build the real torch-native learned geometry module, small but complete, with ablations, observables, and integration hooks.

If a tradeoff appears between simpler and fuller scope, choose the fuller scope unless it would break the typed interface or make smoke tests impossible.

---

## 35. Final P5 Principle

P5 exists to make this question testable:

```text
Does learned context-transport geometry add epistemic modeling power
beyond structured input, auxiliary objectives, and ordinary graph mixing?
```

The answer must remain open.

P5's job is not to make geometry win.  
P5's job is to make geometry real, differentiable, ablatable, gauge-disciplined, and ready for an honest verdict in P6.
