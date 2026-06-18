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
Use JSONL as the canonical Step 1 interchange format and Parquet as an
analytical bridge.

Rationale:
JSONL is inspectable and validates line by line. Parquet supports flat scans and
future Arrow/Polars workflows.

Consequences:
Nested Parquet layout is intentionally deferred. Step 1 stores nested capsule
payloads as canonical `record_json`.

Status:
Accepted.

## ADR-0003: Pydantic as canonical schema layer

Decision:
Use Pydantic v2 models as the canonical HoloCapsule schema.

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
  gauge-invariant geometric observables, and currently ships complete Python
  3.14 GPU/TPU wheels.
- safetensors/AXT decouples the two engines: tensors are compiled once and read
  by either framework, avoiding tight coupling.

Consequences:
- torch enters at the tensorizer/training stage; the schema, substrate, format,
  store, and falsification layers remain dependency-light and framework-free.
- The dependency set grows (torch, JAX/jaxlib, optionally transformers and
  PyTorch Geometric) and CI must cover both engines on Python 3.14. GPU PyTorch
  on 3.14 may require nightly or source builds until official CUDA `cp314`
  wheels are published; the data/format/falsification layers are unaffected.
- Fair-comparison discipline (equal tokens, compute, and parameters across arms)
  is maintained by sharing a standard backbone across arms, so novelty stays in
  the data substrate, the training interface, and the losses.

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
