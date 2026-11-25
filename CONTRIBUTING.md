# Contributing to Axiom

Thanks for your interest. Axiom is a research codebase, so small and focused pull requests are much easier to review than large ones.

## Getting started

```sh
uv python install 3.14
uv sync --python 3.14 --extra dev
./bin/axiom --help
```

Axiom targets Python 3.14 only. The code uses PEP 695 `type` aliases and PEP 758 `except A, B:` syntax, which are valid on 3.14.

## Before you open a PR

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src` and `uv run pytest` pass. CI runs the same four checks from `uv.lock`.
- New behaviour has a test in `tests/`, preferably against the small fixtures in `tests/fixtures/` or `examples/`.
- Read [docs/concept/CONCEPT.md](docs/concept/CONCEPT.md) and [docs/concept/DECISIONS.md](docs/concept/DECISIONS.md). If your change crosses one of the core rules (no truth labels, no temporal leakage, ablatable geometry, providers only for data construction), open an issue first and propose an ADR.
- Format changes (AXF, AXC, AXP, AXT, AXC-out) need a matching update in `spec/` and the examples.
- No API keys, private corpora, account IDs or personal paths in code, configs or fixtures. Use `*.example` files and environment variables.
- Dependency bumps change `pyproject.toml` and `uv.lock` together.

## Commit messages

Use short, imperative subjects ("Add relation mask to AXT compiler"), following the `feat:` / `fix:` / `docs:` / `chore:` prefixes already used in the history. Add a body when the reason isn't obvious.

## Reporting bugs and ideas

Use the issue templates. For security problems, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

By contributing, you agree that your contributions are licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
