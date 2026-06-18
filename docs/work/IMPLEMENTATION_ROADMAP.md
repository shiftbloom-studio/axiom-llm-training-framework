# Axiom v1 — Implementation Roadmap (Checkpoint / Source of Truth)

Last updated: 2026-06-18 · Target: full first version, no reductions, not over-fine-tuned

## Purpose

This is the single source of truth for completing Axiom to a full first version. **If the
working conversation is lost, continue from here.** It defines the six implementation
plans, their execution order, the locked decisions and guardrails, and links to the
governing documents. Each plan is later expanded into its own plan + agent-prompt file in
this folder and handed to one agent.

Concept & decisions: [../concept/CONCEPT.md](../concept/CONCEPT.md) · Visual: [build_map.svg](build_map.svg) · torch/JAX split: [../concept/torch_jax_interface.svg](../concept/torch_jax_interface.svg)

## Where the project stands

- **Built and green on Python 3.14** (95 tests, ruff, mypy): Steps 1–4 — schema/data
  contract, substrate builder (intentional rule-based bootstrap extraction), AXF/AXC/AXP
  formats, torch-free falsification prep. Full audit: [PROGRESS_REVIEW.md](PROGRESS_REVIEW.md).
- **Not yet built:** everything downstream of the training bridge — the real model,
  training loop, learned geometry, and model-based evaluation. The six plans close that gap.
- **Goal:** build the *full theoretical model* properly but small (~10–50M params), with
  real components and sane defaults. Test the real thing; do not run the verdict on a
  reduced stand-in. Do not fine-tune for everything before the first release.

## Locked framework decisions (do not re-litigate)

- PyTorch is primary for the model/training and arms A–D. In-loop **learned geometry is
  torch-native** (PyTorch Geometric); **JAX is reference/precompute only**; a JAX↔torch
  bridge is adopted only if profiling forces it. — [../concept/DECISIONS.md](../concept/DECISIONS.md) ADR-0007, ADR-0009
- Geometry/HKR is **in-plan and developed concurrently** (not optional), but stays
  **ablatable and gauge-invariant**. — ADR-0005, ADR-0008
- **Falsification-first:** equal tokens/compute/params across arms; redundancy ≠
  popularity; only gauge-invariant observables; negative results preserved. —
  [../concept/RESEARCH_PROTOCOL.md](../concept/RESEARCH_PROTOCOL.md), [../concept/FALSIFICATION_HARNESS.md](../concept/FALSIFICATION_HARNESS.md), [../concept/DATA_CONTRACT.md](../concept/DATA_CONTRACT.md)
- Project guidelines/ADRs were Codex-generated and are overrulable by the owner; if a plan
  would need to cross one, ask first.

## First-class must-haves (requirements CLOSED — no additions after this)

1. **Learned geometry as a real trained module** (not precomputed features).
2. **Full epistemic objective set + heads** (not narrowed to next-token + relation).
3. **Relation-neighborhood conditioning**, and keep the **n-ary hypergraph** path open.
4. The epistemic **"router" escalation** (epistemic state steering routing / loss-masking)
   named and kept, not dropped.

## The six plans

### P1 · Substrate & corpus  📄 plan written ([PLAN_1_substrate.md](PLAN_1_substrate.md))
Upgrade extraction to credible claim/relation quality and real epistemic signals; generate
synthetic views (FAQ / teaching note / counterargument / historical update); assemble the
first curated corpus (50–500 claim families) and add PDF ingestion. Corpus domain: **ML/software benchmark claims**.
- Carries first-class: **synthetic views**; also produces gold material feeding P6.
- Refs: WP3; [../concept/CLAIM_FIELD_SUBSTRATE_BUILDER.md](../concept/CLAIM_FIELD_SUBSTRATE_BUILDER.md); [../project-knowledge/HoloCapsule_Project_Knowledge.md](../project-knowledge/HoloCapsule_Project_Knowledge.md) §6–§7, §10; code: `src/hcaps/{ingest,extraction,substrate}`.
- Depends on: Steps 1–4.

### P2 · Data interface & tensorizer  ☐ not started
Real subword tokenizer (BPE/WordPiece) behind the existing protocol; compile capsules/arms
into **AXT** tensor bundles (token ids, masks, loss masks, side-channel tensors,
relation/provenance tensors, split + manifest hashes); relation-**neighborhood sampler**.
- Carries first-class: **relation-neighborhood sampling**.
- Refs: WP4; [../../spec/AXT_TENSOR_BUNDLE.md](../../spec/AXT_TENSOR_BUNDLE.md), [../../spec/AXF.md](../../spec/AXF.md); [../concept/TRAINING_BRIDGE.md](../concept/TRAINING_BRIDGE.md); code: `src/hcaps/training_bridge`.
- Depends on: P1. · Plan file: `PLAN_2_data_interface.md` (TBD)

### P3 · Model & conditioning  ☐ not started
Torch backbone LM + the four arms (A flat-text, B structured-text, C capsule-serialized,
D capsule + side-channel adapter) + auxiliary heads (relation / provenance / stability /
uncertainty). Side-channel adapter/prefix injects capsule features. Name the **router**
escalation (epistemic state steering routing/loss-masking) as a later stage.
- Carries first-class: **router escalation**, **full heads**.
- Refs: WP4–WP6; [../project-knowledge/HoloCapsule_Claim_Field_Pretraining_Architecture.md](../project-knowledge/HoloCapsule_Claim_Field_Pretraining_Architecture.md) §4 (escalation stages); [../project-knowledge/HoloCapsule_Project_Knowledge.md](../project-knowledge/HoloCapsule_Project_Knowledge.md) §11; code: new `src/hcaps/model`.
- Depends on: P2 (P4 concurrent). · Plan file: `PLAN_3_model.md` (TBD)

### P4 · Learned geometry (torch-native)  ☐ not started
Claim-field graph + a learnable **connection** → parallel transport → holonomy / curvature
/ spectrum, exposed only as **gauge-invariant observables**, trained end-to-end with the
model. Keep the **n-ary** relation path open. JAX serves as an off-loop reference for the
math; adopt a live bridge only if profiling forces it.
- Carries first-class: **learned geometry as a trained module**, **n-ary path**.
- Refs: WP9; [../project-knowledge/HoloCapsule_Project_Knowledge.md](../project-knowledge/HoloCapsule_Project_Knowledge.md) §9; [../concept/DATA_CONTRACT.md](../concept/DATA_CONTRACT.md) (Geometry Rules); [../concept/HKR_TO_AXF_MAPPING.md](../concept/HKR_TO_AXF_MAPPING.md); [../concept/DECISIONS.md](../concept/DECISIONS.md) ADR-0005/0008/0009.
- Depends on: P3 interface (concurrent with P3). · Plan file: `PLAN_4_geometry.md` (TBD)

### P5 · Training loop & objectives  ☐ not started
Optimizer / scheduler / seed control / checkpointing + the full multi-term loss
(next-token + future-summary + relation prediction + provenance recovery + uncertainty
calibration + context consistency), every auxiliary term ablatable, under **equal
token/compute/param** budgets across arms.
- Carries first-class: **full epistemic objective set**.
- Refs: WP5; [../project-knowledge/HoloCapsule_Project_Knowledge.md](../project-knowledge/HoloCapsule_Project_Knowledge.md) §12; [../project-knowledge/HoloCapsule_Claim_Field_Pretraining_Architecture.md](../project-knowledge/HoloCapsule_Claim_Field_Pretraining_Architecture.md) §12 (principles); code: new `src/hcaps/training`.
- Depends on: P3, P4. · Plan file: `PLAN_5_training.md` (TBD)

### P6 · Evaluation & falsification + decision  ☐ not started
Epistemic metrics (relation prediction, support/contradiction, provenance recovery,
uncertainty calibration, temporal-stability, outdated-belief QA) + gold sets; train and
score every arm (A–H) with the existing controls (context-shuffle, geometry-off,
no-provenance, no-relations, popularity); emit the **proceed / redesign / kill** report.
- Carries first-class: exercises all controls; the honest verdict on the thesis.
- Refs: WP7–WP8; [../concept/EVALUATION_PROTOCOL.md](../concept/EVALUATION_PROTOCOL.md), [../concept/FALSIFICATION_HARNESS.md](../concept/FALSIFICATION_HARNESS.md), [../concept/RESEARCH_PROTOCOL.md](../concept/RESEARCH_PROTOCOL.md); code: extends `src/hcaps/falsification`.
- Depends on: P5 (and P1 gold sets). · Plan file: `PLAN_6_evaluation.md` (TBD)

## Execution order (phases)

No calendar dates — this is dependency order. Mark each plan done in the checklist above.

- **Phase A — data foundation:** P1 → P2
- **Phase B — model (concurrent):** P3 + P4
- **Phase C — training:** P5
- **Phase D — verdict:** P6

P1 also feeds P6 (gold sets). See [build_map.svg](build_map.svg).

## How each plan becomes an agent prompt

When a plan is started, its file contains: (1) a short **project kick-off** (the context in
this roadmap + [../concept/CONCEPT.md](../concept/CONCEPT.md)), (2) that plan's **scope**, (3) the **locked
decisions + guardrails** above, and (4) the instruction: *fully implement your part — real
components, sane defaults, no reductions, no fine-tuning; keep your first-class must-haves
first-class; wire into the falsification controls.*

## Status at a glance

| Stage | Status |
|---|---|
| Steps 1–4 (schema, substrate, formats, falsification prep) | ✅ done, green on 3.14 |
| P1 Substrate & corpus | 📄 plan written ([PLAN_1](PLAN_1_substrate.md)) |
| P2 Data interface & tensorizer | ☐ not started |
| P3 Model & conditioning | ☐ not started |
| P4 Learned geometry | ☐ not started |
| P5 Training loop & objectives | ☐ not started |
| P6 Evaluation & falsification + decision | ☐ not started |

## Reference map

- Concept & decisions: [../concept/CONCEPT.md](../concept/CONCEPT.md) · [../concept/DECISIONS.md](../concept/DECISIONS.md)
- Vision / source material: [Architecture](../project-knowledge/HoloCapsule_Claim_Field_Pretraining_Architecture.md) · [Project Knowledge](../project-knowledge/HoloCapsule_Project_Knowledge.md) · [Execution Plan](../project-knowledge/HoloCapsule_Execution_Plan.md)
- Contracts & specs: [DATA_CONTRACT](../concept/DATA_CONTRACT.md) · [AXF/AXC/AXP/AXT specs](../../spec/AXF.md)
- Evaluation & falsification: [EVALUATION_PROTOCOL](../concept/EVALUATION_PROTOCOL.md) · [FALSIFICATION_HARNESS](../concept/FALSIFICATION_HARNESS.md) · [RESEARCH_PROTOCOL](../concept/RESEARCH_PROTOCOL.md)
- Training bridge: [TRAINING_BRIDGE](../concept/TRAINING_BRIDGE.md) · HKR mapping: [HKR_TO_AXF_MAPPING](../concept/HKR_TO_AXF_MAPPING.md)
- Current-state review: [PROGRESS_REVIEW.md](PROGRESS_REVIEW.md)
