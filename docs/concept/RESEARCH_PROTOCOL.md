# Research Protocol

Status: Current | Updated: 2026-06-18 | See: [CONCEPT.md](CONCEPT.md), [DECISIONS.md](DECISIONS.md)

Axiom is a structured-native LLM training framework. It tests whether claim-field structure improves epistemic competence under fair controls. It does not assume that the hypothesis is true.

## Research Posture

- Build the full structured-native concept at small scale.
- Keep structured input, structured supervision, architecture changes, and learned geometry separately ablatable.
- Preserve negative results.
- Do not claim performance without an experiment report.
- Do not collapse Axiom into token-in/token-out CausalLM training unless an explicit rollback ADR accepts that path.

## Current Baseline

The repository currently contains foundation infrastructure, not the v1 model or verdict. Legacy Steps 1-4 are substrate and falsification-preparation work. Step 5 is not complete without actual model training, metrics, and a decision report.

## Fairness

Primary fairness:

```text
same source content
same temporal cutoffs
same splits
same extraction substrate
matched parameter budget
matched compute budget / FLOPs
matched training schedule where applicable
```

Token parity applies only within text-rendered arms and text-projection comparisons.

## Provider Use

External or local LLM-compatible providers may be used only for data construction/substrate harvesting.

Provider-backed construction must be:

- cached;
- hashed;
- replayable;
- manifest-backed;
- config-driven;
- ablatable;
- provider-identity tracked.

Provider identity and slant are lateral context. Record provider metadata and feed the constructed substrate equally to all relevant arms.

External LLMs must not be used inside model training, evaluation, scoring, or benchmark judging unless a future ADR explicitly isolates and authorizes that use.

## No Truth Labels

Axiom records claim-state, not binary truth. It may record support, contestation, uncertainty, stability, provenance, and evaluation references. It must not encode `truth`, `is_true`, `correct`, `is_correct`, or equivalent fields as labels.

## Temporal Discipline

Every predictor-side source timestamp must be at or before the valid cutoff. Future-facing fields may exist only as masked targets.

Time is not lateral context. Do not merge temporal drift and lateral context into a single context embedding.

## Geometry Discipline

Geometry is experimental, ablatable, and gauge-invariant. Report only gauge-invariant observables. Do not claim that geometry proves HKR or objectivity.

## Interpreter Discipline

AXC-out evaluation must preserve:

```text
raw_emission
validated_axc_out
interpreted_projection
text_projection
```

The interpreter must be deterministic and ablatable. It must not hide raw model failures.
