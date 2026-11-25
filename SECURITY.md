# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Use GitHub's private vulnerability reporting (Security tab → "Report a vulnerability") or email **safety@shiftbloom.studio** with "axiom security" in the subject. Include steps to reproduce and the impact you see.

We aim to acknowledge reports within 3 working days and to ship a fix or mitigation within 14 days for confirmed issues. We'll credit you in the release notes unless you'd rather stay anonymous.

## Supported versions

Axiom is a research framework without stable releases yet. Only the latest `main` is supported.

## Scope

In scope:

- the optional operator UI in `ui/` (for example SSRF through the fetch endpoints, path traversal into run directories, or anything that exposes provider API keys);
- leaking API keys into run artifacts, provider caches or manifests;
- unsafe deserialization when loading AXT bundles, checkpoints or provider caches from a run directory.

Out of scope: the quality of trained models, and problems that require an attacker who can already edit your configs or run commands on your machine. Note that PyTorch checkpoints (`*.pt`) can execute code when loaded, so only load checkpoints you created yourself.
