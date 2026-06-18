# Corpus Provider Ingress

Axiom P1 adds a provider-aware corpus construction path for the claim-field
substrate. Providers are permitted only for data construction and substrate
harvesting. They are never part of model training, evaluation, scoring, or
benchmark judging.

The ingress flow is:

```text
raw sources
  -> normalized source documents
  -> provider-backed extraction traces
  -> claim candidates and families
  -> relation candidates
  -> epistemic proxies
  -> structured views
  -> AXC streams and AXP packages
```

Provider outputs are cached under `.cache/axiom/providers/` using request and
public provider fingerprints. Cache records store raw response payloads,
normalized outputs, hashes, warnings, and provider traces. API key values and
secret material are never stored.

Supported provider types:

- deterministic bootstrap provider;
- OpenAI-compatible provider for local or remote construction services;
- Python-callable provider for custom local extractors;
- config-driven cascade with deterministic gate and merge traces.

The P1 corpus builder emits source registries, provider trace JSONL,
disagreement records, claim-family records, relation candidates, negative pools,
gold-reference candidates, reports, AXC streams, and AXP packages.

Text views are projections and comparison material. AXC/AXP remain structured
claim-field artifacts, and later AXT/AXC-out work starts in P2/P7.
