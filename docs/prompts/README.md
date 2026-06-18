# Provider Prompt Registry

These templates are versioned contracts for data construction providers. They
are not training prompts and are not used in model evaluation or benchmark
judging.

Provider outputs must be cached, hashed, replayable, provenance-tracked, and
fed equally to downstream arms. Providers must preserve temporal metadata
separately from lateral context and must not emit binary labels.

Current templates:

- `claim_extraction_v0_1.md`
- `relation_extraction_v0_1.md`
- `epistemic_extraction_v0_1.md`
- `view_generation_v0_1.md`
