# AXF v0.1: Axiom Exchange Format

Status: v0.1 compatibility reference. The current v1 semantic contract is
[AXF_V1.md](AXF_V1.md), with P3 handoff details in
[../docs/work/P2_HANDOFF_TO_P3.md](../docs/work/P2_HANDOFF_TO_P3.md).

AXF, the Axiom Exchange Format, is the public format family for claim-centric
LLM pretraining data. AXF represents claim-states rather than documents,
instruction examples, or binary truth labels.

AXF v0.1 includes:

- AXC: Axiom Capsule Stream, line-oriented claim-state records.
- AXP: Axiom Package, a complete dataset package with manifests, splits, and
  controls.
- AXT: Axiom Tensor Bundle, a later compiled training tensor artifact specified
  now and implemented in a future training bridge step.
- AXC-out: the structured model emission format, specified as a reserved v1
  contract and implemented in later model/decoder work.

## Principles

AXF records describe:

```text
claim-family -> context-specific claim-state -> evidence/revision/provenance trajectory
```

The atomic record is a Claim-State Capsule. It separates:

- claim identity;
- claim state;
- surface expression;
- source spans;
- provenance;
- temporal scope;
- context;
- epistemic state;
- relations;
- training eligibility.

AXF does not encode eternal axioms or binary truth labels. It stores observed
stabilization, revision, contradiction, provenance, redundancy, uncertainty, and
context state.

## Formats

| Name | Extension | Purpose |
|---|---|---|
| AXC | `.axc` | Claim-state capsule stream |
| AXP | `.axp/` | Dataset package directory |
| AXT | `.axt`, `.axt.safetensors` | Future compiled tensor bundle |
| AXC-out | `.axc-out`, `.axc-out.json` | Future structured model emission |

Compressed `.axc.zst` and `.axp.tar.zst` are reserved in the
specification but not implemented in v0.1.

AXC v0.1 is internally UTF-8 newline-delimited canonical JSON, but the public
file suffix is `.axc`. JSONL is the encoding strategy; AXC is the format
contract.

## Temporal Policy

AXF is temporal by design. Every AXC capsule must include `temporal.valid_as_of`
and `temporal.constructed_at`. Predictor-side source dates later than
`valid_as_of` are invalid unless the source or span is explicitly marked
`target_only`.

## Epistemic Policy

AXF stores epistemic proxies such as evidential anchoring, uncertainty,
transformation pressure, and independent redundancy. It does not store fields
such as `truth`, `is_true`, or `correct`.

Independent redundancy is an effective independence/diversity quantity, not raw
citation count or popularity.

## Geometry Policy

AXF reserves a `geometry` section for future context-transport observables.
Canonical AXC records allow only gauge-invariant observables. Raw connection
matrices are not stable semantic AXF fields.

The canonical gauge policy is:

```text
gauge_invariant_observables_only
```

## AXT Policy

AXT must include both predictor-side input tensors and target/output tensors.
It must include explicit loss masks, target availability, temporal masks, and
negative-sampling metadata for relation, provenance, context, temporal,
near-but-distinct, and same-topic unrelated samples.

Missing targets are masked. They are not negative labels.

## AXC-out Policy

AXC-out must preserve:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

Raw emissions are scored directly. The interpreter must be deterministic,
versioned, ablatable, and unable to hide malformed model output.

## Falsification Controls

AXP packages reserve manifest locations for temporal holdouts, context shuffles,
degree-preserving rewires, popularity/recency controls, embedding-only
baselines, relation ablations, geometry-disabled ablations, and
provenance-disabled ablations.
