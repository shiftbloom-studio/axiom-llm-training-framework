# Axiom Documentation

This directory contains the active documentation set for Axiom.

Axiom is a structured-native LLM training framework. The canonical concept, decisions, roadmap, and specs are the authority for future work. Historical source material is preserved for provenance, but it is not allowed to override the active Axiom direction.

## Start Here

1. [concept/CONCEPT.md](concept/CONCEPT.md) - conceptual authority.
2. [concept/DECISIONS.md](concept/DECISIONS.md) - accepted ADRs.
3. [work/IMPLEMENTATION_ROADMAP.md](work/IMPLEMENTATION_ROADMAP.md) - current P1-P6 v1 roadmap.
4. [../spec/AXF_V1.md](../spec/AXF_V1.md) - AXF v1 semantic contract.
5. [concept/CORPUS_PROVIDER_INGRESS.md](concept/CORPUS_PROVIDER_INGRESS.md) - P1 corpus/provider ingress.
6. [concept/AXT_RUNTIME_DATA_INTERFACE.md](concept/AXT_RUNTIME_DATA_INTERFACE.md) - P3 AXT compiler/runtime interface.
7. [MODEL_STACK.md](MODEL_STACK.md) - P4 structured-native model stack.
8. [GEOMETRY_MODULE.md](GEOMETRY_MODULE.md) - P5 learned claim-field geometry.
9. [TRAINING_RUNTIME.md](TRAINING_RUNTIME.md) - P6 training runtime.
10. [EXPERIMENT_ORCHESTRATOR.md](EXPERIMENT_ORCHESTRATOR.md) - P6 experiment arms and controls.
11. [SCORING_AND_VERDICT.md](SCORING_AND_VERDICT.md) - P6 scoring and branch verdicts.
12. [OPERATOR_RUNTIME.md](OPERATOR_RUNTIME.md) - P6 run inspection and export.
13. [P6_HANDOFF_FINAL.md](P6_HANDOFF_FINAL.md) - final P6 handoff.
14. [work/P5_HANDOFF_TO_P6.md](work/P5_HANDOFF_TO_P6.md) - geometry handoff consumed by P6.
15. [work/P4_HANDOFF_TO_P5.md](work/P4_HANDOFF_TO_P5.md) - P4 geometry hook handoff consumed by P5.
16. [work/P4_HANDOFF_TO_P6.md](work/P4_HANDOFF_TO_P6.md) - model/loss handoff.
17. [work/P3_AXT_COMPILER_HANDOFF_TO_P4.md](work/P3_AXT_COMPILER_HANDOFF_TO_P4.md) - P3 handoff consumed by P4/P5/P6.
18. [work/P2_HANDOFF_TO_P3.md](work/P2_HANDOFF_TO_P3.md) - P2 handoff consumed by P3.
19. [work/PRE_P1_DOCUMENTATION_AUDIT.md](work/PRE_P1_DOCUMENTATION_AUDIT.md) - record of the pre-P1 alignment pass.
20. [work/POST_P6_HARDENING_AUDIT.md](work/POST_P6_HARDENING_AUDIT.md) - record of the post-P6 hardening pass.
21. [REVIEWER_GUIDE.md](../REVIEWER_GUIDE.md) - concise post-P6 reviewer orientation + reproduction steps.
22. [SPONSOR_ONE_PAGER.md](../SPONSOR_ONE_PAGER.md) - sponsor one-pager (markdown source + visual asset in assets/).

## Active Folders

| Folder | Role |
|---|---|
| [concept/](concept/) | Current concept, ADRs, protocol docs, architecture notes, and guardrails. |
| [work/](work/) | Current roadmap and current work/audit records. |
| [../spec/](../spec/) | AXF, AXC, AXP, AXT, and AXC-out specifications. |
| [../examples/](../examples/) | Small AXF/AXC/AXP conformance fixtures. |
| [../configs/](../configs/) | Smoke configs for corpus, AXT, model, geometry, training, experiments, scoring, and verdict runtimes. |

## Historical Material

| Folder | Role |
|---|---|
| [project-knowledge/](project-knowledge/) | Historical source material and research notes. These files may use old HoloCapsule terminology. |
| [archive/](archive/) | Deprecated point-in-time plans, status reports, and diagrams superseded by the current roadmap. |

Historical files may contradict current terminology. When they do, follow:

1. [concept/CONCEPT.md](concept/CONCEPT.md)
2. [concept/DECISIONS.md](concept/DECISIONS.md)
3. [work/IMPLEMENTATION_ROADMAP.md](work/IMPLEMENTATION_ROADMAP.md)
4. [../spec/](../spec/)

## Current Next Action

```text
Review docs/work/POST_P6_HARDENING_AUDIT.md and choose the next research-readiness or external-review work item.
```

Readiness materials for that choice:
- `docs/REVIEWER_GUIDE.md`
- `docs/SPONSOR_ONE_PAGER.md` + `docs/assets/axiom-sponsor-onepager.jpg`

Do not start new work from archived Plan 1/Plan 2 documents. They are historical drafts from the old roadmap shape.
