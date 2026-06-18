# Claim-Field Substrate Builder

The claim-field substrate builder is the first real data-construction layer in
Axiom. It converts local source documents into validated claim-centric capsules
that can later feed tokenizers, collators, and training objectives.

This component is production-oriented infrastructure, but the current extractor
is intentionally simple. It is a deterministic bootstrap extractor, not a
scientific claim-understanding system.

## Purpose

Flat pretraining pipelines usually reduce documents to token streams. Axiom keeps
claim identity, provenance, temporal state, relations, and uncertainty visible.
The builder turns:

```text
source files -> source documents -> chunks -> candidate claims -> claim families
-> relation candidates -> validated claim capsules -> manifest-tracked JSONL
```

The output capsules validate against the Step 01 schema. Legacy internal schema
names such as `HoloCapsule` remain for backward compatibility, but public-facing
builder docs and CLI use neutral terms such as `ClaimCapsule`, `ClaimField`, and
`Substrate`.

## Supported Formats

The reader supports:

- `.txt`
- `.md`
- `.markdown`
- `.jsonl` records with a `text` field

PDF input is detected and skipped with a clear warning in Step 02. It can be
added later through a lightweight reader if Python 3.14-compatible dependencies
are selected.

For each source document, the builder records:

- source path and file name;
- media type;
- raw byte SHA-256 hash;
- normalized text SHA-256 hash;
- text length;
- title, authors, license, source URL, and domains when provided;
- publication date and temporal cutoff when provided;
- reader name and version.

## Sidecar Metadata

Optional sidecars use the same base name:

```text
paper.md
paper.meta.json
```

Supported strict fields:

```json
{
  "title": "Example Source",
  "authors": ["Researcher"],
  "publication_date": "2026-01-01",
  "license": "CC-BY-4.0",
  "source_url": "https://example.invalid/source",
  "domain": ["science"],
  "temporal_cutoff": "2026-01-01"
}
```

Invalid sidecars produce warnings by default. The builder can be configured to
treat them as errors.

## Chunking

Chunking is deterministic and prefers semantic boundaries:

- Markdown headings define section paths;
- blank lines define paragraph boundaries;
- sentence boundaries split overlong paragraphs;
- hard character splitting is used only as a fallback.

Chunk IDs are content-addressed with a `chunk_` prefix and include document ID,
chunk index, offsets, and text.

## Bootstrap Claim Extraction

The default extractor is a deterministic local rule system. It:

- splits chunks into sentences;
- rejects short, long, rhetorical, and boilerplate sentences;
- selects assertive sentences with simple cues;
- classifies weak cue categories such as causal, method, empirical, definition,
  comparison, limitation, and temporal;
- assigns conservative confidence values in `[0.0, 1.0]`;
- preserves source offsets and section context.

This extractor is replaceable through the `ClaimExtractor` protocol. Future
backends may be neural, human-curated, or LLM-assisted, but they should emit the
same `CandidateClaim` contract so downstream grouping and capsule construction do
not change.

## Claim Families

Candidate claims are canonicalized by Unicode normalization, whitespace
normalization, lowercase fingerprints, punctuation trimming, and tokenization.
Near-equivalent candidates are grouped using deterministic token/edit similarity.

Each family records:

- `family_` ID;
- schema-compatible `claim_` ID;
- canonical claim text;
- member claim IDs;
- source document IDs;
- domains;
- first-seen date when known;
- evidence count;
- independent source count proxy;
- confidence summary.

## Relation Candidates

Relation candidates use conservative local heuristics. Supported relation types
align with the capsule schema:

- `supports`
- `contradicts`
- `extends`
- `supersedes`
- `mentions`
- `related`

The heuristics use lexical overlap plus cue phrases such as "supports",
"contradicts", "fails to replicate", "replaces", and "mentions". Confidence is
intentionally conservative. Undocumented certainty is not introduced.

## Temporal Leakage Guard

The CLI supports a build-level cutoff:

```bash
axiom build-substrate \
  --input docs/source-material \
  --output data/claim-field/capsules.jsonl \
  --manifest data/claim-field/build_manifest.json \
  --cutoff-date 2026-01-01
```

When a source has a publication date after the cutoff, strict mode excludes it
and records a `post_cutoff_source_excluded` warning in the build manifest.
Undated sources cannot be temporally excluded; they are retained with explicit
unknown-date provenance behavior.

## Deterministic IDs

Persisted research artifacts use SHA-256-derived stable IDs:

- `doc_`
- `src_`
- `chunk_`
- `claim_`
- `family_`
- `rel_`
- `cap_`

The emitted capsule prefix remains `cap_` because the Step 01 schema requires it.

## Manifest

Every build writes a JSON manifest containing:

- run ID;
- package and pipeline version;
- Python version;
- timestamp;
- input and output paths;
- config hash;
- source, chunk, claim, family, relation, capsule, and skipped counts;
- warnings;
- input file hashes;
- output capsule file hash;
- temporal cutoff.

The manifest does not include a self-hash for itself because that would be
self-referential. The capsule output is content-hashed.

## CLI

Build capsules:

```bash
axiom build-substrate \
  --input tests/fixtures/source_docs \
  --output data/claim-field/capsules.jsonl \
  --manifest data/claim-field/build_manifest.json \
  --cutoff-date 2026-01-01
```

Inspect a capsule JSONL:

```bash
axiom inspect-substrate --input data/claim-field/capsules.jsonl
```

The CLI fails clearly on missing input, creates output directories, validates
capsules through the existing JSONL store, and prints a concise summary.

## Current Limitations

- The bootstrap extractor is transparent and testable, but shallow.
- Relation candidates are heuristic and low-confidence.
- PDF input is skipped in Step 02.
- Undated sources cannot be rigorously placed in time.
- Epistemic state fields are explicit proxies, not truth labels.
- No training loop, neural extractor, external LLM call, or benchmark claim is
  introduced in this step.
