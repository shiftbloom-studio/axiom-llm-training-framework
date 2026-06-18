# Decisions

## ADR-0001: Python first

Decision:
Use Python 3.14+ as the primary implementation language.

Rationale:
Python has the strongest ecosystem for data validation, Arrow/Parquet tooling,
scientific experiments, and later ML training integration.

Consequences:
Performance-critical paths may later move to Rust, C++, CUDA, or Triton only
after profiling proves a bottleneck.

Status:
Accepted.

## ADR-0002: JSONL plus Parquet as initial storage formats

Decision:
Use newline-delimited canonical JSON as the initial internal capsule stream
encoding and Parquet as an analytical bridge. Public canonical capsule streams
use the `.axc` suffix under the AXC contract.

Rationale:
JSONL is inspectable and validates line by line. Parquet supports flat scans and
future Arrow/Polars workflows.

Consequences:
Nested Parquet layout is intentionally deferred. The legacy internal store may
still use `.jsonl`; public AXC examples and AXP packages should use `.axc`.

Status:
Accepted.

## ADR-0003: Pydantic as canonical schema layer

Decision:
Use Pydantic v2 models as the canonical claim-state schema layer. The current
internal model name `HoloCapsule` remains for compatibility; public project
identity is Axiom.

Rationale:
Pydantic provides strict runtime validation, JSON parsing, typed models, and good
error messages for fixture and storage validation.

Consequences:
All stores and future builders must emit records that pass the Pydantic contract.

Status:
Accepted.

## ADR-0004: No model training in Step 1

Decision:
Do not implement model training in Step 1.

Rationale:
The project must first stabilize the data contract and storage substrate.

Consequences:
No torch dependency, no tokenizer bridge, no benchmark claims, and no training
artifacts are added in this step.

Status:
Accepted.

## ADR-0005: HKR geometry is experimental and ablatable

Decision:
Represent HKR/geometric values only as optional experimental gauge-invariant
summaries.

Rationale:
The project tests whether HKR-inspired structure helps. It does not assume it is
true or necessary.

Consequences:
Raw gauge matrices are excluded, geometry can be removed from any capsule, and
future evaluations must include ablations.

Status:
Accepted.

## ADR-0006: Temporal leakage prevention is a schema-level concern

Decision:
Validate source and target timestamps in the schema.

Rationale:
Target leakage should fail before training data reaches tensorization.

Consequences:
Capsules reject model-visible source timestamps after the cutoff and
future-facing target timestamps at or before the cutoff.

Status:
Accepted.

## ADR-0007: PyTorch primary training framework; JAX for geometry/HKR numerics

Decision:
Adopt PyTorch as the primary training and modeling framework for the training
bridge, the model arms (flat-text, structured-text, capsule-serialized, and
capsule side-channel), auxiliary heads, and the training loop. Adopt JAX as a
committed framework for the geometry/HKR numerics in the claim-field geometry
workstream (see ADR-0008), including curvature, holonomy, and spectrum
observables and their custom autodiff. Use the AXT tensor bundle
(`.axt.safetensors`) as the framework-neutral interchange boundary so both
engines consume the same compiled tensors. TensorFlow is not adopted.

Rationale:
- PyTorch gives the most direct path to Hugging Face-compatible backbones,
  adapter/prefix conditioning, PyTorch Geometric relation graphs, and rapid
  ablation. It is the training stack named in the source-material plan.
- JAX provides first-class jacobians, `vmap`, and custom autodiff well suited to
  gauge-invariant geometric observables.
- safetensors/AXT decouples the two engines: tensors are compiled once and read
  by either framework, avoiding tight coupling.

Consequences:
- torch enters at the tensorizer/training stage; the schema, substrate, format,
  store, and falsification layers remain dependency-light and framework-free.
- The dependency set grows (torch, JAX/jaxlib, optionally transformers and
  PyTorch Geometric) and CI must cover both engines on Python 3.14. GPU PyTorch
  on 3.14 may require nightly or source builds until official CUDA `cp314`
  wheels are published; the data/format/falsification layers are unaffected.
- Refined by ADR-0011: shared stock-backbone fairness applies only to text-only
  baselines and rollback experiments. The v1 target is a structured-native
  custom torch model. Primary fairness is matched source content, temporal
  cutoffs, splits, extraction substrate, parameter budget, compute/FLOPs, and
  schedule where applicable. Token parity applies only inside text-rendered arms
  and text-projection comparisons.

Status:
Accepted.

## ADR-0008: Geometry/HKR and claim-field graph workstream is in-plan and concurrent

Decision:
Treat the claim-field graph and HKR-inspired geometry workstream (the work
package previously framed as optional) as a committed, in-plan part of the
program, and develop it concurrently with the core capsule arms rather than
only after a clean capsule result.

Rationale:
- The program intends to test the geometry branch as a first-class hypothesis,
  and the falsification infrastructure already reserves geometry-disabled
  ablation and context-shuffle controls for it.
- Concurrent development shortens the path to evaluating the geometry branch
  alongside the core capsule-vs-text comparison.

Consequences:
- Geometry/HKR remains experimental and ablatable per ADR-0005: observables stay
  gauge-invariant, geometry can be disabled, and every geometric result must be
  checked against a parameter-matched non-geometric baseline and a
  context-shuffle control (see `PROJECT_KNOWLEDGE.md` "HKR and Geometry",
  `DATA_CONTRACT.md` "Geometry Rules", and `FALSIFICATION_HARNESS.md`).
  Committing this workstream changes its delivery status, not its scientific
  status.
- Because geometry is built concurrently rather than after a clean capsule
  result, experiment design must keep the capsule-vs-flat/structured comparison
  separable from the geometry arm so the first result is not confounded; the
  geometry-disabled control must always be run.
- JAX is the committed numerics engine for this workstream (ADR-0007).

Status:
Accepted.

## ADR-0009: In-loop learned geometry is torch-native; JAX is reference/precompute

Decision:
Implement the in-loop, end-to-end learned geometry (the learnable connection and its
gauge-invariant observables) natively in PyTorch (PyTorch Geometric plus custom modules),
in a single autograd graph with the model. Use JAX only off the training critical path:
to prototype and validate the geometry math and for offline (Mode A) precompute. Adopt a
live JAX <-> PyTorch bridge only if the torch-native geometry profiles as a bottleneck.

Rationale:
- The thesis depends on geometry being learnable end-to-end; a single autograd graph is
  the smallest-surface-area way to guarantee gradients reach the connection parameters.
- `torch.func` (`vmap`/`jacrev`/`jacfwd`) plus `torch.compile` cover the jacobian and
  vectorization ergonomics that previously motivated using JAX inside the loop.
- Keeping JAX off the critical path avoids a cross-framework VJP/DLPack bridge as a
  default dependency.

Consequences:
- Refines ADR-0007: PyTorch owns the in-loop learned geometry; JAX's committed role
  narrows to reference implementation and offline precompute.
- The bridge remains an option, adopted reactively after profiling (mirrors the
  "only after profiling proves a bottleneck" rule in ADR-0001).
- Geometry stays ablatable and gauge-invariant per ADR-0005 and ADR-0008.

Status:
Accepted.

## ADR-0010: Data-ingress provider interface + local-first cascade; LLMs for data construction only

Decision:
Introduce a single universal **OpenAI-API-compatible provider interface** (`base_url` +
`model` + key) for model-backed data ingress/extraction, usable with a local or remote
model interchangeably, plus a generic Python-callable provider. The interface keeps local and remote models first-class and swappable, and includes a
**local-first cascade**: a configured local model runs the bulk pass, an escalation gate
routes low-confidence / high-impact items to a remote model, then results merge. The
cascade is **config-driven and ablatable** (gate off ⇒ single provider). Caching + hash +
replay reproducibility is retained. Strike the
earlier "no external LLM API" rule and replace it with: LLMs may be used for **data
construction / substrate harvesting**, never in the **training or evaluation** path.

Rationale:
- Keeps the framework-not-solution seam (any provider plugs in). This is where earlier work
  drifted: the fix is a clean boundary between *how raw data/intelligence enters* and *the
  extraction logic*.
- The local-first cascade is low extra effort (a second endpoint + a gate + merge,
  config-driven) and routes only the ~10–20% hard/high-impact items to remote — most of the
  quality where it matters, at a fraction of the cost.
- The produced substrate is pinned and content-hashed and feeds **all arms equally**, so
  LLM-built data does not bias the arm comparison.

Consequences:
- Supersedes the Step-1 "no external LLM" guideline (see RESEARCH_PROTOCOL.md update).
- Provider config (`base_url`/`model`) selects local or remote with no code change; both are
  first-class, with the deterministic extractor as offline fallback.
- The **local-first cascade** is part of scope: bulk local → gated remote escalation → merge,
  all config-driven and **ablatable** (gate off ⇒ single provider). Each item records which
  provider produced it (cross-extractor signal).
- Caching + hash + replay reproducibility and full extraction
  (claims/relations/epistemic/views) are retained — no scope reduction.
- Tie-breaker when refining: keep the fuller scope (slightly more complex), not less.

Status:
Accepted.

## ADR-0011: Structured-native model I/O (text is a secondary projection)

Decision:
The v1 model is **structured-native**: its primary input *and* output are the Axiom
structure (claim-graph + epistemic / provenance / context vectors + gauge-invariant
geometry), not a token stream. It ingests structured representations (encode-in), computes
at full complexity (learned geometry + epistemic routing — core), and **emits a structured,
interpretable Axiom output ("AXC-out")** that a decoder/interpreter renders. **Text is a
secondary projection**: a text-decoder head produces tokens from the same internal state, so
the model stays comparable to flat-text baselines and scorable on perplexity.

Rationale:
- Realizes Concept §2/§3 + decision 11: full complexity from input to output, no plain-text
  boundary; raising *both* endpoints is the lever for the higher knowledge ceiling.
- The text projection is the comparability bridge — it keeps the falsification arms (flat /
  structured text) meaningful without capping the model at token-in / token-out.
- Owner choice (most radical / highest potential), consistent with ADR-0008 + decision 10
  (no reduction; lean more complex).

Consequences (refines ADR-0007/0009):
- **Backbone:** a custom torch structured model (structured encoder + full-complexity core +
  structured decoder + text-projection head + interpreter), not a stock HF causal LM.
  PyTorch stays primary; torch-native learned geometry sits in the core; HF is optional, only
  for the text projection / baselines. JAX stays reference/precompute.
- **New output artifact:** a structured emission format ("AXC-out") + an interpreter/decoder;
  AXT (input) gains a matching **target/output tensor spec**; AXC-out traces back to inputs +
  manifest hashes. P2 specifies the output-target contract; P7/P8/P9 build the
  decoder, structured-primary objective, and structured metrics.
- **Fairness re-grounded:** arms are matched on **equal parameters + compute + source
  content**; token-budget parity applies **within the text arms / projection**. The
  structured-native arm is scored on epistemic tasks *and* on its text projection. Controls
  unchanged (structure-off / text-only = baseline; geometry-off; context-shuffle; popularity).
- Highest-effort variant chosen deliberately. A text-only configuration remains as an honest
  **rollback** if C proves intractable — a fallback, not the target.

Status:
Accepted.

## ADR-0012: Canonical roadmap reset to v1 implementation programs

Decision:
Replace the older six-plan roadmap and legacy Step-completion vocabulary with
the ten-program v1 roadmap in `docs/work/IMPLEMENTATION_ROADMAP.md`:

1. P1 Claim-Field Corpus & Provider Ingress
2. P2 AXF v1 Contract, AXT & AXC-out Specification
3. P3 AXT Compiler & Runtime Data Interface
4. P4 Structured Encoder & Relation/Hypergraph Conditioning
5. P5 Learned Geometry Core
6. P6 Full-Complexity Core & Epistemic Router
7. P7 Structured Decoder, AXC-out Interpreter & Text Projection
8. P8 Multi-Objective Training Runtime & Curriculum
9. P9 Experiment Arm Orchestrator & Decision Runtime
10. P10 Open Research Runtime & Operator Surface

Rationale:
The old six-plan roadmap mixed infrastructure, model work, training, geometry,
and evaluation in ways that now contradict the structured-native concept and
ADR-0011. The v1 work needs explicit AXT, AXC-out, structured encoder,
learned geometry, epistemic router, structured decoder, training runtime, and
decision-runtime programs.

Consequences:
- Legacy Steps 1-4 remain historical foundation/infrastructure.
- Legacy Step 3 is a lightweight training bridge, not P3.
- Legacy Step 4 prepares falsification artifacts, not the evaluation verdict.
- Legacy Step 5 is not complete without actual training/evaluation artifacts and
  a decision report.
- Archived Plan 1/Plan 2 drafts are not active implementation plans.
- The next action after the pre-P1 documentation alignment pass is to create
  Implementation Plan P1: Claim-Field Corpus & Provider Ingress.

Status:
Accepted.
