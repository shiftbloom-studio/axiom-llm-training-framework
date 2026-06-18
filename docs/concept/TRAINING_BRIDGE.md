# Training Bridge

Status: Current | Updated: 2026-06-18 | See: [DATA_CONTRACT.md](DATA_CONTRACT.md), [../../spec/AXT_V1_TENSOR_BUNDLE.md](../../spec/AXT_V1_TENSOR_BUNDLE.md)

The current training bridge is a lightweight infrastructure layer. It renders claim-state capsules into text comparison arms, tokenizes them with a baseline tokenizer, builds side-channel vectors, and records manifests.

It is not the full v1 AXT compiler, model runtime, or training loop.

## Current Modes

Current implemented renderers:

- `flat_text` - claim surface only, for text baseline comparison;
- `structured_text` - labeled text view with claim, context, and sources;
- `capsule_text` - richer predictor-visible capsule rendering.

These are text-rendered arms. They are useful for baselines and text projection. They are not the structured-native Axiom substrate.

## Future AXT Compiler

P3 must implement the production AXT compiler from [../work/P2_HANDOFF_TO_P3.md](../work/P2_HANDOFF_TO_P3.md). AXT must include:

- input tensors;
- target/output tensors;
- loss masks;
- target availability masks;
- temporal masks;
- relation/neighborhood tensors;
- provenance tensors;
- context tensors;
- geometry-observable tensors where enabled;
- deterministic negative-sampling metadata;
- split metadata and manifest hashes.

The compiler must prevent future-facing targets from entering predictor-visible tensors.

## Tokenization

Tokenization supports text-rendered arms and text projection. It should not become the native representation by accident.

The current `WhitespaceTokenizer` is a baseline/fallback. Future text arms may use a real subword tokenizer, but token parity applies only inside text-rendered arms and text-projection comparisons.

## Side Channels

Current side-channel features are numeric projections of epistemic state, relation counts, provenance counts, and geometry flags. They are useful infrastructure but do not replace structured-native tensors.

Future AXT should make side channels ablatable and preserve the relation/hypergraph path directly.

## Manifests

Every bridge or compiler artifact should record:

- input paths and hashes;
- tokenizer identity and hash where used;
- rendering mode or structured arm;
- sequence length or tensor shape limits;
- side-channel inclusion;
- split identity;
- deterministic seed;
- output hashes.

## Guardrails

- no truth labels;
- no temporal leakage;
- missing targets require masks;
- deterministic negative samples where targets need negatives;
- no raw gauge matrices as canonical outputs;
- no hidden provider or interpreter repair.
