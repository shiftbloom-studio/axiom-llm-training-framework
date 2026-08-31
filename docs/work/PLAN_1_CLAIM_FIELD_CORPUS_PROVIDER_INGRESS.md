# PLAN 1 / 10 — Claim-Field Corpus & Provider Ingress

**Repository:** `shiftbloom-studio/axiom-llm-training-framework`  
**Branch suggestion:** `feat/p1-claim-field-corpus-provider-ingress`  
**Expected commit message:** `feat: implement claim-field corpus and provider ingress`  
**Status:** Implementation plan, ready for agent execution  
**Scope class:** All-or-nothing implementation program  
**Roadmap position:** P1 of 10  

---

## 0. Executive intent

P1 turns Axiom's current substrate builder from a deterministic bootstrap into a credible, provider-aware corpus construction system for the first real claim-field corpus.

The goal is not to train a model. The goal is to produce the first high-quality, replayable, provenance-rich, temporally scoped AXF/AXC/AXP substrate that future AXT compilation and structured-native model training can consume.

P1 must keep Axiom's current identity intact:

> Axiom is a structured-native LLM training framework. It stays LLM-compatible through text projection and text baselines, but its primary substrate is claim-field structure, not a flat token stream.

This plan implements the ingress side of that identity:

```text
raw sources
  -> normalized source documents
  -> provider-backed extraction traces
  -> claim candidates
  -> claim families
  -> relation candidates
  -> epistemic/provenance/context features
  -> synthetic structured views
  -> curated first corpus
  -> AXC streams + AXP packages + audit manifests
```

P1 is the first step where Axiom starts to become more than a format/schema project. It creates the practical research substrate for P2-P10.

---

## 1. Governing concept and decisions

Agents implementing this plan must follow the current concept and ADR direction.

### 1.1 Structured-native posture

Axiom is not a text-only CausalLM wrapper. P1 must produce structured data that later compiles into AXT and supports AXC-out. Text renderings may exist, but they are projections and baselines, not the native substrate.

### 1.2 Data construction providers are allowed

LLMs and other model-backed providers may be used for **data construction / substrate harvesting** only. They must never enter the training or evaluation/scoring path.

Provider output must be:

- cached;
- content-hashed;
- replayable;
- provenance-tracked;
- provider-attributed;
- fed equally to all downstream experiment arms;
- ablatable by disabling provider escalation or replacing providers.

### 1.3 Local-first cascade

P1 must implement the provider ingress interface in a way that supports:

```text
bulk local provider
  -> confidence / impact / uncertainty gate
  -> remote escalation for selected hard cases
  -> deterministic merge
  -> cache + trace + provenance
```

The cascade must be config-driven and ablatable. Turning the escalation gate off must yield a single-provider path.

### 1.4 Time is not lateral context

P1 must keep temporal metadata separate from lateral context metadata.

Temporal axis:

- publication date;
- observed-at date;
- valid-as-of cutoff;
- target-only/future-facing markers;
- temporal holdout boundaries.

Lateral context axis:

- source community;
- provider/extractor identity;
- field/subfield;
- benchmark family;
- task area;
- paper/repository/document type;
- language/register;
- framing/claim formulation.

Do not collapse time into a generic context string.

### 1.5 No truth labels

P1 must never emit binary truth labels.

Forbidden fields in active AXF/AXC artifacts include:

```text
truth
is_true
correct
is_correct
factuality
ground_truth
label_truth
proven_true
```

When a human-curated evaluation reference is required, use names such as:

```text
evaluation_reference
gold_reference
human_verified_reference
```

Those references must describe evaluation targets, not metaphysical truth.

### 1.6 Geometry remains reserved

P1 may collect context and relation material useful for later geometry, but it must not claim to learn or validate HKR geometry.

If geometry-related metadata is emitted, it must be clearly marked as preliminary context/graph material, not learned curvature or proof.

---

## 2. Plan objective

Implement the first full corpus-ingress program for Axiom:

> Build a provider-aware, replayable, local-first claim-field substrate generator and use it to create the first curated ML/software benchmark claim corpus as AXC/AXP artifacts.

The implementation must upgrade the existing substrate path without breaking existing AXF/AXC/AXP conformance.

---

## 3. Non-goals

Do not implement:

- AXT tensor compilation;
- AXC-out model output runtime;
- structured-native model architecture;
- learned geometry module;
- training loop;
- optimizer/scheduler/checkpointing;
- benchmark claims;
- model evaluation verdicts;
- Step/P2-P10 implementation plans.

Do not make claims like:

```text
Axiom improves LLMs.
Axiom proves HKR.
This corpus validates structured-native training.
```

P1 creates substrate. It does not decide the thesis.

---

## 4. Required deliverables

### 4.1 Provider ingress layer

Create a provider abstraction under a coherent package path, preferably:

```text
src/hcaps/providers/
  __init__.py
  base.py
  config.py
  cache.py
  openai_compatible.py
  python_callable.py
  deterministic.py
  cascade.py
  merge.py
  traces.py
  errors.py
```

If the repo already has a provider package, extend it rather than duplicating it.

Required provider types:

1. **DeterministicProvider**
   - wraps the existing deterministic extraction path;
   - always available;
   - used as offline fallback and baseline.

2. **OpenAICompatibleProvider**
   - accepts `base_url`, `model`, `api_key_env`, timeout, max retries;
   - compatible with local servers and remote providers;
   - must not hard-code OpenAI as the only provider;
   - must support dry-run / cache-only operation.

3. **PythonCallableProvider**
   - accepts a Python import path such as `module:function`;
   - used for custom local extraction functions;
   - useful for non-LLM extractors or experimental heuristics.

4. **ProviderCascade**
   - runs bulk extraction with a primary provider;
   - applies a gate;
   - escalates selected records to secondary provider;
   - merges outputs deterministically;
   - emits trace metadata.

Provider configuration must support local and remote models without code changes:

```yaml
providers:
  primary:
    type: openai_compatible
    base_url: http://localhost:8000/v1
    model: local-extractor-model
    api_key_env: AXIOM_LOCAL_PROVIDER_KEY
  escalation:
    type: openai_compatible
    base_url: https://api.example.invalid/v1
    model: remote-high-quality-model
    api_key_env: AXIOM_REMOTE_PROVIDER_KEY
  cascade:
    enabled: true
    gate:
      min_confidence: 0.72
      high_impact_claim_types:
        - causal_claim
        - measurement_claim
      max_escalation_fraction: 0.2
```

### 4.2 Provider cache and replay

Implement a content-addressed cache.

Suggested path:

```text
.cache/axiom/providers/
  <provider_fingerprint>/
    <request_hash>.json
```

The cache key must include at least:

- provider type;
- provider base URL fingerprint, not secret;
- model name;
- prompt/template version;
- request payload hash;
- schema/output contract version;
- extraction task type;
- relevant config hash.

Cache records must include:

```json
{
  "cache_version": "0.1.0",
  "request_hash": "sha256:...",
  "provider": {
    "type": "openai_compatible",
    "base_url_fingerprint": "sha256:...",
    "model": "..."
  },
  "task": "claim_extraction",
  "created_at": "...",
  "prompt_version": "...",
  "input_hash": "sha256:...",
  "output_hash": "sha256:...",
  "raw_response": {...},
  "normalized_output": {...},
  "warnings": []
}
```

Do not store API keys or secrets.

Modes:

```text
live        use provider, write cache
cache_only  never call provider, fail if missing cache
refresh     call provider even if cache exists, write new cache record
```

### 4.3 Extraction task contracts

Define stable input/output contracts for provider-backed extraction. Use Pydantic models.

Suggested package:

```text
src/hcaps/extraction/contracts.py
```

Required extraction task outputs:

1. `ClaimExtractionOutput`
   - candidate claims;
   - claim type;
   - confidence;
   - source span references;
   - reasoning/notes optional but not model-evaluated;
   - provider metadata.

2. `RelationExtractionOutput`
   - source claim;
   - target claim;
   - relation type;
   - confidence;
   - evidence spans;
   - candidate hard negatives if available;
   - provider metadata.

3. `EpistemicExtractionOutput`
   - evidential anchoring proxy;
   - ontology/ontic compatibility proxy;
   - transformation pressure proxy;
   - uncertainty proxy;
   - independent redundancy estimate/proxy;
   - confidence and method fields.

4. `StructuredViewOutput`
   - FAQ view;
   - teaching note;
   - counterargument;
   - limitation view;
   - historical update / temporal note if source material supports it;
   - neutral summary;
   - technical summary.

5. `ProviderTrace`
   - provider ID;
   - provider type;
   - provider model;
   - extraction task;
   - prompt/template version;
   - cache key;
   - input hash;
   - output hash;
   - confidence;
   - escalation reason;
   - merge decision.

### 4.4 Prompt/template registry

If provider-backed extraction uses prompts, create a versioned prompt/template registry.

Suggested path:

```text
docs/prompts/
  README.md
  claim_extraction_v0_1.md
  relation_extraction_v0_1.md
  epistemic_extraction_v0_1.md
  view_generation_v0_1.md
```

or, if prompts should be code-side:

```text
src/hcaps/providers/templates/
```

Prompt templates must explicitly instruct providers:

- do not output truth labels;
- output claim-state and uncertainty only;
- preserve source-span grounding;
- distinguish temporal facts from lateral context;
- do not invent sources;
- mark uncertain fields as unknown/null;
- output JSON only;
- use schema-compatible enums when possible.

### 4.5 First corpus domain: ML/software benchmark claims

Create a corpus-building configuration for ML/software benchmark claims.

Suggested directory:

```text
configs/corpus/ml_software_benchmarks/
  ingress.yaml
  providers.local.yaml.example
  providers.remote.yaml.example
  extraction.yaml
  corpus.yaml
```

The initial corpus should be small enough to inspect but real enough to exercise the pipeline.

Target initial corpus size:

```text
50-500 claim families
```

Preferred source categories:

- ML benchmark papers;
- model technical reports;
- software engineering benchmark reports;
- dataset papers;
- reproducibility/replication notes;
- benchmark cards or leaderboards if license permits;
- repository README/release notes when used as source material;
- evaluation methodology documents.

The corpus must record license and source restrictions. Do not silently use restricted sources.

The repository should include tiny fixture sources, not a huge real corpus. Large or license-sensitive corpora should be generated locally by users or stored outside Git with manifests.

### 4.6 PDF ingestion

Existing docs say PDF is detected and skipped. P1 should add real PDF text ingestion if feasible under Python 3.14.

Requirements:

- dependency-light;
- deterministic;
- page-level metadata where possible;
- clear fallback when PDF parsing fails;
- page number attached to spans/chunks;
- no OCR by default;
- OCR not introduced unless explicitly configured in a later plan.

Acceptable libraries if compatible:

- `pypdf` or comparable lightweight parser.

If PDF support would introduce heavy or unstable dependencies, implement a plugin interface and a clear not-enabled error, but keep the interface ready.

### 4.7 Source registry and corpus manifest

Add a source registry layer if missing.

Suggested package:

```text
src/hcaps/corpus/
  __init__.py
  registry.py
  manifest.py
  builder.py
  sampling.py
  quality.py
  gold.py
```

The corpus manifest should include:

```json
{
  "corpus_id": "axiom.ml_software_benchmark_claims.v0",
  "corpus_version": "0.1.0",
  "created_at": "...",
  "domain": ["machine_learning", "software_engineering", "benchmarking"],
  "source_count": 0,
  "claim_family_count": 0,
  "provider_policy": {...},
  "license_summary": {...},
  "temporal_policy": {...},
  "hashes": {...},
  "warnings": []
}
```

Every source file must have:

- stable ID;
- raw hash;
- normalized hash;
- license;
- publication/retrieval dates where available;
- source URL/path;
- source type;
- provider/extractor provenance;
- temporal cutoff policy.

### 4.8 Synthetic structured views

P1 must generate structured views for each claim family when enough source material exists.

Required views:

```text
neutral_summary
technical_summary
teaching_note
faq
counterargument
limitations
historical_update_or_temporal_note
```

These are not training labels by default. They are structured text projections and view material that later arms may use.

Each generated view must record:

- provider/extractor;
- prompt/template version;
- source spans used;
- confidence;
- warnings;
- cache key;
- whether the view is human-reviewed.

### 4.9 Epistemic proxy extraction

P1 must produce explicit but conservative epistemic proxies.

Required proxy families:

```text
ontology_compatibility / ontic_compatibility

evidential_anchoring
transformation_pressure
uncertainty
independent_redundancy
```

Important rules:

- Values are proxies, not truth.
- Redundancy is not raw popularity.
- Raw citation/mention/source counts may be recorded as separate simple counts, but they must not be named `independent_redundancy`.
- Each proxy must record method and confidence.
- Unknown/unsupported proxy values should be missing/null or confidence-low, not invented.

Suggested additional fields:

```text
source_count
provider_disagreement_count
relation_degree
method_diversity_proxy
source_type_diversity_proxy
benchmark_family_diversity_proxy
```

These can later support popularity/frequency controls.

### 4.10 Provider disagreement and merge traces

The provider cascade must not hide disagreement.

If local and remote providers disagree, record:

```text
local_output_hash
remote_output_hash
disagreement_type
merge_policy
merged_output_hash
selected_fields
rejected_fields
human_review_required flag
```

Provider disagreement is lateral context and may later become useful signal. Do not flatten it away.

### 4.11 Human-review hooks and gold set

P1 must add hooks for a small human-reviewed gold/reference set, even if the actual set is tiny.

Suggested files:

```text
data/gold/README.md
configs/corpus/ml_software_benchmarks/gold_schema.yaml
examples/gold/ml_software_claims_gold.jsonl
```

Gold references should support:

- claim boundary review;
- relation review;
- provenance span review;
- temporal cutoff review;
- epistemic proxy review where feasible;
- hard negative relation examples;
- near-but-distinct claim examples.

Avoid the word `truth`.

Use:

```text
human_verified_reference
evaluation_reference
gold_reference
```

### 4.12 Negative material for later training/evaluation

P1 must generate or at least preserve negative candidate pools for later P2/P5/P6.

Required negative pools:

- relation hard negatives;
- same-topic unrelated claims;
- near-but-distinct claims;
- provenance distractor sources;
- temporal distractors;
- provider-disagreement cases.

Do not train on them in P1. Store them in corpus artifacts/manifests.

### 4.13 CLI commands

Add or extend CLI commands under a `corpus` or `ingress` group.

Suggested commands:

```bash
axiom corpus build \
  --config configs/corpus/ml_software_benchmarks/corpus.yaml

axiom corpus inspect \
  data/corpus/ml_software_benchmarks.axp

axiom corpus providers check \
  --config configs/corpus/ml_software_benchmarks/providers.local.yaml

axiom corpus cache inspect \
  .cache/axiom/providers

axiom corpus gold export \
  --input data/corpus/ml_software_benchmarks.axp \
  --output data/gold/ml_software_candidates.jsonl
```

If the existing CLI architecture prefers a different grouping, use it consistently.

### 4.14 Output artifacts

P1 must produce or support producing:

```text
artifacts/corpus/ml_software_benchmarks/
  corpus_manifest.json
  provider_cache_manifest.json
  extraction_trace.jsonl
  provider_disagreements.jsonl
  source_registry.jsonl
  claim_families.jsonl
  relation_candidates.jsonl
  negative_pools.jsonl
  gold_candidates.jsonl
  leakage_report.json
  license_report.json
  quality_report.json
  dataset.axp/
    axiom.json
    data/
      capsules.axc
      sources.axsrc
      relations.axr
      contexts.axctx
    splits/
      train.json
      validation.json
      temporal_holdout.json
    manifests/
      files.json
      hashes.json
      construction_report.json
      leakage_report.json
      provider_report.json
      license_report.json
```

The exact path may differ, but the information must exist.

---

## 5. Required repository changes

### 5.1 New or extended packages

Implement or extend:

```text
src/hcaps/providers/
src/hcaps/corpus/
src/hcaps/extraction/contracts.py
src/hcaps/provenance/       # if not already sufficient
src/hcaps/ingest/readers.py # PDF and provider-aware source metadata
src/hcaps/substrate/        # integrate provider outputs
src/hcaps/format/           # only if AXP manifests need provider reports
src/hcaps/cli.py            # corpus/ingress commands
```

Do not create a parallel system that bypasses existing `ingest`, `extraction`, `substrate`, `format`, or `store` layers.

### 5.2 Configs

Add:

```text
configs/corpus/ml_software_benchmarks/corpus.yaml
configs/corpus/ml_software_benchmarks/providers.local.yaml.example
configs/corpus/ml_software_benchmarks/providers.remote.yaml.example
configs/corpus/ml_software_benchmarks/extraction.yaml
configs/corpus/ml_software_benchmarks/gold.yaml
```

### 5.3 Documentation

Add:

```text
docs/work/PLAN_1_CLAIM_FIELD_CORPUS_PROVIDER_INGRESS.md

docs/CORPUS_PROVIDER_INGRESS.md
or

docs/concept/CORPUS_PROVIDER_INGRESS.md
```

Update:

```text
README.md
AGENTS.md if necessary
IMPLEMENTATION_ROADMAP.md status marker only after completion
CLAIM_FIELD_SUBSTRATE_BUILDER.md
DATA_CONTRACT.md if new metadata is added
AXF/AXP docs only if provider manifest surfaces are formalized
```

### 5.4 Examples and fixtures

Add minimal examples:

```text
examples/corpus/ml_software_benchmarks/
  README.md
  sources/
    benchmark_claims.md
    benchmark_claims.meta.json
  configs/
    deterministic_only.yaml
  expected/
    minimal_corpus_manifest.json
```

Keep fixtures tiny and synthetic or clearly license-safe.

### 5.5 Tests and quality checks

Although roadmap steps are not separate validation phases, every implementation plan must include its own checks.

Add tests for:

- provider config parsing;
- deterministic provider behavior;
- OpenAI-compatible provider dry-run/cache-only behavior;
- cache key stability;
- cache replay;
- cascade gating;
- deterministic merge;
- provider trace serialization;
- no secrets in cache;
- no forbidden truth labels;
- provider metadata preserved;
- sidecar metadata integration;
- PDF reader behavior or graceful skip/plugin error;
- synthetic view generation schema;
- epistemic proxy extraction schema;
- negative pool creation;
- corpus manifest hashing;
- AXP package creation;
- CLI smoke path.

Run:

```bash
ruff check .
ruff format .
mypy src
pytest
```

If external-provider tests would require network, mark them as offline mocks only.

---

## 6. Architecture details

### 6.1 Provider abstraction

Suggested type model:

```python
class ProviderRequest(BaseModel):
    task: ExtractionTask
    input_text: str
    source_ref: SourceRef
    schema_version: str
    template_version: str
    context: dict[str, Any]


class ProviderResponse(BaseModel):
    provider_trace: ProviderTrace
    normalized_output: dict[str, Any]
    confidence: float | None
    warnings: list[str]


class ExtractionProvider(Protocol):
    provider_id: str

    def run(self, request: ProviderRequest) -> ProviderResponse: ...
```

Provider-specific raw outputs should not leak directly into AXC. They must pass through normalization.

### 6.2 Escalation gate

Suggested gate inputs:

- extraction confidence below threshold;
- high-impact claim type;
- relation ambiguity;
- contradiction/supersession cue;
- provider uncertainty;
- low source grounding;
- high centrality/source count proxy;
- sample fraction limit.

Suggested gate output:

```python
class EscalationDecision(BaseModel):
    escalate: bool
    reason_codes: list[str]
    score: float
```

The gate must be deterministic.

### 6.3 Merge policy

Suggested merge policy order:

1. schema-valid fields beat invalid fields;
2. source-grounded fields beat ungrounded fields;
3. higher-confidence fields beat lower-confidence fields;
4. human-reviewed fields beat provider fields;
5. deterministic fallback resolves remaining ties by provider priority then field hash.

Every merge decision must be traceable.

### 6.4 Provider slant as context

Provider/extractor identity must be preserved as lateral context.

Record at minimum:

```text
provider_id
provider_type
provider_model
provider_family
provider_mode: deterministic | local | remote | human
prompt_template_version
extraction_method
cascade_stage
merge_role
```

This can later support provider-shuffle or provider-context-off controls.

### 6.5 Corpus quality tiers

P1 should classify claim families into quality tiers.

Suggested tiers:

```text
bronze: automatically extracted, schema-valid, not human-reviewed
silver: provider-consensus or high-confidence, source-grounded
gold: human-reviewed evaluation reference
quarantine: schema-valid but warning-heavy / low confidence / temporal ambiguity
```

Quality tier is not truth. It is construction confidence.

### 6.6 First corpus sampling strategy

The initial corpus should not be a random pile of sources.

Construct it to include:

- benchmark performance claims;
- method improvement claims;
- dataset limitation claims;
- reproducibility claims;
- contradiction/failed-replication style claims;
- supersession/update claims;
- same-topic near-duplicate claims;
- temporally ordered claim families where possible.

This matters because later arms need relation and temporal signal.

---

## 7. Data contract expectations

P1 must output AXC-compatible capsules or a clearly versioned compatible extension.

Each capsule must preserve or reference:

- claim identity;
- claim family identity;
- source spans;
- provenance sources;
- temporal valid-as-of information;
- lateral context;
- epistemic proxies;
- relation candidates;
- synthetic views;
- provider traces;
- construction confidence;
- warnings;
- no truth labels.

If the current AXC schema cannot store provider traces cleanly, store them in the AXP construction/provider report and link them by IDs. Do not force unstable metadata into AXC fields unless versioned.

---

## 8. Interaction with later plans

### P2 depends on P1 for:

- claim-family corpus;
- relation candidate pools;
- provider traces;
- negative pools;
- loss-mask source availability;
- AXP manifests;
- initial gold candidates;
- source/context metadata.

### P3/P4 depend on P1 indirectly for:

- relation neighborhoods;
- lateral context metadata;
- source/provider context axes;
- hard negative examples;
- early graph construction.

### P5/P6 depend on P1 for:

- structured targets;
- evaluation references;
- source-grounded provenance recovery examples;
- temporal split material;
- popularity/frequency controls.

P1 should therefore emit enough metadata for later plans even when not used immediately.

---

## 9. Agent execution sequence

This sequence is implementation order inside P1. It is not a separate roadmap.

1. Inspect existing substrate, ingest, extraction, format, CLI, and tests.
2. Add provider contracts and config models.
3. Add provider cache/replay system.
4. Add deterministic provider wrapper around existing extractor.
5. Add OpenAI-compatible provider with cache-only/dry-run-safe design.
6. Add Python-callable provider.
7. Add local-first cascade with deterministic gate and merge traces.
8. Add provider-backed extraction task outputs for claims, relations, epistemic proxies, and structured views.
9. Integrate provider outputs into substrate builder without breaking deterministic mode.
10. Add corpus manifest/source registry layer.
11. Add ML/software benchmark corpus configs and tiny fixtures.
12. Add synthetic view generation and provider trace preservation.
13. Add negative pool and gold-candidate export hooks.
14. Add PDF reader/plugin or graceful enabled/disabled interface.
15. Add CLI commands and documentation.
16. Add tests and run quality gates.
17. Update roadmap status only if P1 is fully implemented and green.

---

## 10. Acceptance criteria

P1 is complete only when all of the following are true.

### Functional criteria

- Provider abstraction exists and supports deterministic, OpenAI-compatible, and Python-callable providers.
- Provider outputs are cached, hashed, replayable, and secret-free.
- Local-first cascade exists and is config-driven/ablatable.
- Provider traces are stored and linked to generated claims/views/proxies.
- Existing deterministic substrate build still works.
- Provider-backed substrate build works in cache-only/mock mode without external network.
- ML/software benchmark corpus config exists.
- Tiny fixture corpus can build an AXC/AXP artifact.
- Synthetic views are emitted or explicitly unavailable with warnings.
- Epistemic proxies are emitted conservatively with method/confidence.
- Negative pools/gold candidates are generated or exported.
- PDF ingestion is implemented or explicitly plugin-gated with clear errors.
- CLI can run the full tiny corpus path.

### Scientific criteria

- No truth labels are introduced.
- Redundancy is not raw popularity.
- Time is separate from lateral context.
- Provider identity is preserved as lateral context.
- Provider disagreement is not discarded.
- All downstream arms would consume the same pinned substrate.
- The produced substrate is suitable input for P2's AXT/AXC-out specification work.

### Engineering criteria

- Python 3.14 compatibility is preserved.
- Existing Step 1-4 tests still pass.
- New tests cover provider/cache/cascade/corpus paths.
- Public docs use Axiom terminology.
- `hcaps` remains only as legacy internal package name if not renamed.
- No network dependency in tests.
- No API keys/secrets are written to disk.
- Large generated artifacts are not committed unless tiny fixtures.

---

## 11. Required prompt for the implementation agent

Use the following prompt when handing P1 to Codex/Opus/GPT agent.

```text
You are implementing P1 / 10 for the Axiom repository.

Repository:
shiftbloom-studio/axiom-llm-training-framework

Plan:
PLAN_1_CLAIM_FIELD_CORPUS_PROVIDER_INGRESS.md

Mission:
Implement Claim-Field Corpus & Provider Ingress.

Axiom is a structured-native LLM training framework. It remains LLM-compatible through text projection and text baselines, but its primary substrate is claim-field structure, not a flat token stream. P1 must upgrade the corpus and data-ingress layer so later plans can compile structured AXT tensors and train a structured-native model.

You must implement the full P1 scope, not a reduced version:
- provider abstraction;
- deterministic provider;
- OpenAI-compatible provider;
- Python-callable provider;
- cache + replay;
- local-first cascade;
- deterministic merge traces;
- provider-backed extraction contracts;
- synthetic views;
- epistemic proxy extraction;
- ML/software benchmark corpus config and fixture path;
- source registry and corpus manifest;
- negative pools and gold-candidate hooks;
- PDF ingestion or plugin-gated PDF interface;
- CLI integration;
- tests and documentation.

Hard constraints:
- No model training.
- No AXT compiler.
- No AXC-out runtime.
- No benchmark claims.
- No truth labels.
- No API calls in tests.
- No secrets in cache or logs.
- LLM providers are allowed only for data construction/substrate harvesting, never training/evaluation.
- Provider output must be cached, hashed, replayable, and fed equally to all downstream arms.
- Time and lateral context must stay distinct.
- Provider identity/slant must be recorded as lateral context.
- Do not collapse the system into text-only capsule serialization.

Implementation policy:
- Extend existing modules instead of creating parallel incompatible systems.
- Keep deterministic mode working.
- Use Python 3.14 and typed Pydantic models where appropriate.
- Keep `hcaps` as the internal legacy package name unless the repository has already renamed it.
- Update docs and configs as part of the implementation.
- Add tests for new behavior and run quality gates.

Quality gates:
ruff check .
ruff format .
mypy src
pytest

Expected branch:
feat/p1-claim-field-corpus-provider-ingress

Expected commit:
feat: implement claim-field corpus and provider ingress
```

---

## 12. Notes for reviewers

Reviewers should focus on whether P1 creates a **better substrate**, not whether it trains a model.

Important review questions:

- Can every provider-generated field be traced back to a source/provider/cache record?
- Can the same substrate be replayed without provider access?
- Are local and remote providers interchangeable?
- Can the cascade be disabled?
- Does the corpus contain relation/temporal/provenance signal, not just isolated claims?
- Are provider disagreements preserved?
- Are there any hidden truth labels?
- Are text views clearly projections rather than native substrate?
- Does the output give P2 enough material for AXT, target masks, relation neighborhoods, and AXC-out specs?

P1 should be accepted only if the answer to all of these is yes.
