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
