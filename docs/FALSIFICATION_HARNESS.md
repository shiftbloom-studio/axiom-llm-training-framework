# Falsification Harness

Step 4 turns the repository's research posture into reproducible artifacts. The
existing docs already set the rules: `DATA_CONTRACT.md` requires strict
capsules, `EVALUATION_PROTOCOL.md` defines strong comparison arms,
`RESEARCH_PROTOCOL.md` says negative results must remain visible,
`PROJECT_KNOWLEDGE.md` separates evidence from popularity, and `DECISIONS.md`
makes temporal leakage prevention and ablatability non-negotiable.

The harness consolidates those rules into a torch-free preparation layer. It
does not train a model, run benchmark claims, or call external LLM APIs. It
produces rendered text arms, derived capsule controls, audits, metrics, hashes,
and manifests that Step 5 can consume.

## Why Step 4 Exists

Axiom should not merely create richer data. It must make the central hypothesis
testable before scale: if structured claim-field examples help later, the effect
must survive boring explanations such as rewritten text, source counts, topic
similarity, popularity, relation degree, or future information slipping into
predictor inputs.

## Baselines

`flat_text` exposes only surface or canonical text. It asks whether ordinary
token streams already carry enough signal.

`structured_text` uses stable headings for claim, context, and sources. It asks
whether any later effect is simply caused by clearer text views rather than
capsule-specific fields.

`capsule_text` renders predictor-visible capsule structure while excluding
future-only training targets. It is the rich text arm that later training can
compare against baselines.

## Ablations

The no-provenance, no-relations, no-context, no-side-channel, and
relation-ablation arms remove one family of information at a time. They make it
possible to detect whether provenance, relations, context, and side channels are
separable signals or merely decorative metadata.

Derived artifacts are marked as noncanonical when the transform intentionally
breaks the strict `HoloCapsule` schema, for example by removing required
provenance or context.

## Shuffle Controls

`context_shuffle` deterministically reassigns context fibers across records. If
later context-aware methods cannot tell shuffled context from original context,
then context labels are not carrying the intended signal.

`provenance_shuffle` deterministically reassigns provenance records while
preserving per-record source counts. This helps distinguish source-count effects
from source-identity effects.

## Popularity And Frequency Controls

The popularity/frequency control preserves simple proxies such as source count,
relation count, and claim-family frequency while removing richer epistemic
structure. This follows the project rule that independent redundancy is not raw
popularity, citation count, view count, or repetition.

## Temporal Leakage

Temporal leakage audit is mandatory. Predictor-visible source timestamps must
not be later than the capsule's valid-as-of cutoff. Future-facing target fields
may exist as training targets, but rendered predictor text must not expose
`future_summary`, `target_timestamp`, `target_only`, `ground_truth`, or similar
markers.

Any temporal leakage finding should block downstream training preparation until
the source artifact is repaired.

## Outputs

Runs write:

```text
artifacts/falsification/<run_id>/
  manifest.json
  audits/
  metrics/
  arms/
```

Each arm has a rendered JSONL artifact and an arm manifest. Derived capsule
artifacts use `.axc` only when the transformed records remain canonical
`HoloCapsule` records; otherwise they use `.jsonl` and are marked
`derived_noncanonical`.

## CLI

Implemented commands:

```bash
axiom falsify run examples/axf/v0_1/minimal_dataset.axp --output-dir artifacts --arms all --seed 13 --allow-all-without-split
axiom falsify audit examples/axf/v0_1/minimal_capsules.axc --json
axiom falsify report artifacts/falsification/<run_id>/manifest.json --output report.md
```

`run` creates artifacts and manifests. `audit` runs only input audits. `report`
creates a Markdown readiness report.

## Step 5 Hand-Off

Step 5 can use these manifests to select comparable training streams, verify
hashes, inspect audit status, and compare dataset diagnostics before any model
work begins. Step 4 does not decide whether Axiom wins. It creates the evidence
trail needed for Step 5 to ask that question fairly.
