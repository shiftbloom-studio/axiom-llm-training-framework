# CLAUDE.md

The full agent guide for this repo is **[AGENTS.md](AGENTS.md)** — read it first. This file
only adds Claude-specific pointers; AGENTS.md is the single source of truth so the two
never diverge.

## Entry points

- Navigation: [docs/README.md](docs/README.md)
- Concept + owner decisions + guardrails: [docs/concept/CONCEPT.md](docs/concept/CONCEPT.md)
- Decisions (ADRs): [docs/concept/DECISIONS.md](docs/concept/DECISIONS.md)
- Live status & next step: [docs/work/IMPLEMENTATION_ROADMAP.md](docs/work/IMPLEMENTATION_ROADMAP.md)

## Reminders

- **Python 3.14 only.** The code uses PEP 695 `type` aliases and PEP 758 parenthesis-free
  `except A, B:` — these are valid on 3.14, not bugs. Run with a 3.14 interpreter
  (`uv python install 3.14`); don't flag or "fix" that syntax.
- **Ask before crossing a guideline/ADR.** They were Codex-generated and are owner-overrulable, but never break (or silently comply with) one without asking.
- **No reductions.** Build the full concept at small scale; keep geometry ablatable and gauge-invariant; never emit truth labels.
- **Do not skip the roadmap gate.** P1 is next, but it starts with a new P1 implementation plan, not archived Plan 1/Plan 2 drafts.
- Keep `ruff check`, `ruff format --check`, `mypy src`, and `pytest` green.
