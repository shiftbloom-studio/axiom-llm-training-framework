# Axiom — Documentation (start here)

Navigation for the docs. Only `README.md` lives outside this folder; everything else is here.

- **Current status & next step →** [work/IMPLEMENTATION_ROADMAP.md](work/IMPLEMENTATION_ROADMAP.md)
- **What we're building & why →** [concept/CONCEPT.md](concept/CONCEPT.md)

## Folders

| Folder | What's in it |
|---|---|
| [`concept/`](concept/) | The current architectural + philosophical-physical concept, the owner's decisions, and the guardrails. **Authoritative for intent.** |
| [`work/`](work/) | Implementation plans (current + older), the timeline, and our current position on it (which plan is next). **Live status.** |
| [`project-knowledge/`](project-knowledge/) | Original HKR theory and the initially-provided research material that has not been overruled. **Historical** — intent, not current state. |
| [`../spec/`](../spec/) | Format specifications: AXC / AXF / AXP / AXT. |

## Key documents

| Document | One-line |
|---|---|
| [concept/CONCEPT.md](concept/CONCEPT.md) | Current concept + owner decisions + guardrails (meaning preserved). Start here for direction. |
| [concept/DECISIONS.md](concept/DECISIONS.md) | Formal decision record (ADRs 0001–0009). |
| [concept/ARCHITECTURE.md](concept/ARCHITECTURE.md) | System architecture (predates Steps 2–4; see CONCEPT + ROADMAP for current). |
| [concept/DATA_CONTRACT.md](concept/DATA_CONTRACT.md) | The capsule data contract. |
| [work/IMPLEMENTATION_ROADMAP.md](work/IMPLEMENTATION_ROADMAP.md) | The 6-plan roadmap, build order, and current position. |
| [work/build_map.svg](work/build_map.svg) | Visual of the 6 plans and their order. |
| [work/PLAN_1_substrate.md](work/PLAN_1_substrate.md) | First implementation plan + agent prompt. |
| [work/PROGRESS_REVIEW.md](work/PROGRESS_REVIEW.md) | Point-in-time code audit (2026-06-18): Steps 1–4 done, Step 5 not yet. |

## Reading order (newcomer or agent)

1. This file. 2. [concept/CONCEPT.md](concept/CONCEPT.md). 3. [work/IMPLEMENTATION_ROADMAP.md](work/IMPLEMENTATION_ROADMAP.md).
4. [concept/DECISIONS.md](concept/DECISIONS.md). 5. [concept/DATA_CONTRACT.md](concept/DATA_CONTRACT.md) + [../spec/AXF.md](../spec/AXF.md).
6. [work/PROGRESS_REVIEW.md](work/PROGRESS_REVIEW.md). 7. [project-knowledge/](project-knowledge/) (origins).

## Status convention

Each canonical doc should carry a one-line header: `Status: Current | Updated: YYYY-MM-DD | See: …`. (Rollout pending.)

## Local cleanup (the sandbox can rename but not delete)

These need a local `rm` and then a commit:

- ~18 stray duplicate doc/`.pdf` files at the **repo root** (canonical copies now live under `docs/`). Keep only root `README.md`.
- Empty husk dirs left by the reorg: `docs/diagrams/`, `docs/research/`, `docs/_probe/`.
- Throwaway test artifacts: `.uvtest/`, `.reorg_probe2.txt`.

Then commit the new structure (`docs/README.md`, `docs/concept/`, `docs/work/`, renamed `docs/project-knowledge/`).
