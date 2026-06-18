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
7. [work/P3_AXT_COMPILER_HANDOFF_TO_P4.md](work/P3_AXT_COMPILER_HANDOFF_TO_P4.md) - P3 handoff for P4/P5/P6.
8. [work/P2_HANDOFF_TO_P3.md](work/P2_HANDOFF_TO_P3.md) - P2 handoff consumed by P3.
9. [work/PRE_P1_DOCUMENTATION_AUDIT.md](work/PRE_P1_DOCUMENTATION_AUDIT.md) - record of the pre-P1 alignment pass.

## Active Folders

| Folder | Role |
|---|---|
| [concept/](concept/) | Current concept, ADRs, protocol docs, architecture notes, and guardrails. |
| [work/](work/) | Current roadmap and current work/audit records. |
| [../spec/](../spec/) | AXF, AXC, AXP, AXT, and AXC-out specifications. |
| [../examples/](../examples/) | Small AXF/AXC/AXP conformance fixtures. |
| [../configs/](../configs/) | Smoke configs, corpus configs, and disabled future-runtime config stubs. |

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
Create Implementation Plan P4: Structured-Native Model Stack.
```

Do not start new work from archived Plan 1/Plan 2 documents. They are historical drafts from the old roadmap shape.
