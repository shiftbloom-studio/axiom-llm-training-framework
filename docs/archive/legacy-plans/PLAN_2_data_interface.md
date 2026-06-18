# Historical document. Superseded by CONCEPT.md and IMPLEMENTATION_ROADMAP.md.

This file is an archived draft from the older roadmap shape. It is not an active implementation plan, and P2 must not be started from it.

# Plan 2 — Data Interface & Tensorizer (implementation plan + agent prompt)

Part of [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) · Concept: [../concept/CONCEPT.md](../concept/CONCEPT.md) · Phase A · Status: ☐ not started

---

## Project kick-off (read first)

You are implementing one part of **Axiom / HoloCapsule**, a falsification-first framework
testing whether pretraining on structured **claim-field capsules** beats flat text on
**epistemic competence**. Steps 1–4 are built and green on Python 3.14; Plan 1 turns the
substrate into credible, validated claim-field capsules. **Plan 2 is the bridge from those
capsules to model-ready tensors** — it makes the comparison arms trainable while keeping the
whole experiment fair, reproducible, and leakage-free.

Read for context: [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md),
[../concept/CONCEPT.md](../concept/CONCEPT.md), [../concept/TRAINING_BRIDGE.md](../concept/TRAINING_BRIDGE.md),
[../concept/DATA_CONTRACT.md](../concept/DATA_CONTRACT.md), and the format specs
[../../spec/AXT_TENSOR_BUNDLE.md](../../spec/AXT_TENSOR_BUNDLE.md) + [../../spec/AXF.md](../../spec/AXF.md).

## Design principles (internalize — they shape every task)

1. **AXT is the framework-neutral boundary.** Compile to `.axt` (manifest) +
   `.axt.safetensors` (tensors). safetensors is read by **PyTorch** (the model, Plans 3/5)
   *and* by **JAX** (the geometry reference, Plan 4) — same tensors, no coupling. The
   tensorizer knows nothing about the model.
2. **Tokenizer-agnostic format, one tokenizer in practice.** The AXT format does not fix a
   tokenizer (per the AXT spec's non-goals); it **records the tokenizer identity + hash**.
   We choose one real subword tokenizer so all arms share a vocabulary.
3. **Fair comparison is enforced here.** All arms (A flat-text, B structured-text, C
   capsule-serialized, D capsule + side-channel) are tensorized under an **equal token
   budget** with the **same tokenizer and special-token IDs**. Differences between arms must
   be representation only — never tokenizer, budget, or split.
4. **No leakage.** Future-only fields (`future_summary`, `target_timestamp`, target spans)
   go into **target/label tensors and objective masks only**, never into predictor-visible
   input tensors. Enforce and test this.
5. **Traceable + reproducible.** Every AXT bundle is content-hashed and carries the source
   AXC/AXP manifest hashes, tokenizer hash, arm, budget, and split hashes. Deterministic
   given a seed.
6. **Keep the thesis-carrying structure open.** Use a **relation layout that does not
   hard-code binary edges** (incidence-style, so n-ary hyperedges are addable later) and
   **reserve an optional gauge-invariant geometry tensor slot** (empty here; filled by
   Plan 4). Reserving is cheap; removing it would reduce scope.

## What this plan delivers (scope)

Turn validated claim-field capsules (AXC, optionally inside AXP) into **AXT tensor bundles**
for every comparison arm, plus the **real tokenizer**, the **relation-neighborhood sampler**,
the side-channel/epistemic/provenance tensors, the **objective masks**, and a streaming
dataset/collator that yields real tensors. Carries the first-class must-have
**relation-neighborhood conditioning** and keeps the **n-ary path** open.

## What to build

1. **Real subword tokenizer** (`src/hcaps/training_bridge/tokenizer.py`, extend). Add a
   BPE/WordPiece tokenizer (Hugging Face `tokenizers`) behind the existing `TokenizerProtocol`:
   train on the corpus *or* load a pretrained vocab; persist vocab/merges; deterministic.
   Define and freeze the **special tokens** shared by all arms — `pad`, `unk`, `bos`, `eos`,
   plus structural markers for the structured/capsule arms (`[CLAIM]`, `[CTX]`, `[REL:type]`,
   `[PROV]`, `[SUMMARY]`, `<CAPSULE …>`). Keep `WhitespaceTokenizer` as a baseline/fallback.
   Record tokenizer id + content hash for the manifest.
2. **AXT format I/O** (`src/hcaps/format/axt.py`, new). Write/read `.axt` manifest +
   `.axt.safetensors`. Implements the AXT v0.1 intended contents
   ([../../spec/AXT_TENSOR_BUNDLE.md](../../spec/AXT_TENSOR_BUNDLE.md)): `token_ids`,
   `attention_mask`, `loss_mask`/`labels`, `capsule_ids`, `claim_family_ids`, relation
   tensors, provenance tensors, epistemic scalar tensors, context ids/embeddings, objective
   masks, split metadata, and source AXF/AXP manifest hashes. Validates back to AXC/AXP.
3. **Tensorizer** (`src/hcaps/training_bridge/tensorize.py`, new). Compile capsules → arrays
   for a chosen **arm** using the existing renderers in `examples.py`:
   - **Text + masks:** tokenize the arm's rendered text; build `attention_mask`, next-token
     `labels`, and a **future-summary target span** with its own loss mask (target-only).
   - **Relation tensors (n-ary-capable):** an **incidence layout** — `(relation_id, type_id,
     member_idx, role, confidence)` — so binary edges work now and n-ary hyperedges need no
     schema change later.
   - **Provenance tensors:** per-capsule source count, source-id indices, source timestamps
     (`≤ cutoff`), evidence type — for the provenance-recovery objective.
   - **Epistemic scalar tensors:** bounded scalars (`uncertainty`, `evidential_anchoring`,
     `transformation_pressure`, `ontic_compatibility`, `redundancy_effective_n`) + a
     `stability` one-hot.
   - **Side-channel tensor:** the fixed-dim float vector from `side_channels.py`; record the
     feature order; optional recorded normalization; ablatable (zeroable).
   - **Objective masks:** which positions feed which loss (next-token, future-summary,
     relation, provenance, stability) — the contract Plan 5 consumes.
   - **Reserved slots:** optional `context_embedding` and an optional **gauge-invariant
     geometry** tensor (empty here; Plan 4 fills).
4. **Relation-neighborhood sampler** (`src/hcaps/training_bridge/neighborhoods.py`, new) —
   *first-class*. Deterministic, seeded **k-hop** sampling of the claim-graph around each
   capsule (typed edges, capped fan-out), emitting neighbor index/type tensors for
   conditioning. Incidence-based so it extends to n-ary. No leakage across the temporal
   cutoff (don't sample future-only neighbors into inputs).
5. **Arm parity + equal budget.** A small arm-compilation entrypoint that tensorizes A–D
   from the same capsules under one **token budget** and one tokenizer; record the budget and
   per-arm token counts in the manifest; fail if arms diverge on tokenizer/budget/split.
   Reuse the falsification arm definitions so AXT arms line up with `src/hcaps/falsification`.
6. **Dataset + collator** (`src/hcaps/training_bridge/dataset.py`, `collator.py`, extend).
   Read **AXT** bundles (memory-mapped / shardable, not whole-file-in-memory); deterministic
   seeded shuffling; collator pads and stacks into real **tensors** (torch via safetensors),
   returns `input_ids`, `attention_mask`, `labels`, `loss/objective masks`, side-channel,
   relation/neighbor, provenance, epistemic tensors, and ids. Keep AXC/AXP as accepted inputs
   upstream of tensorization.
7. **Manifest + housekeeping.** AXT manifest records tokenizer hash, arm, seq length + token
   budget, split hashes, source AXC/AXP hashes, counts, seed, build-config hash. Add the
   missing `src/hcaps/training_bridge/__init__.py`; replace deprecated `datetime.utcnow()`
   (`training_bridge/manifests.py`) with `datetime.now(UTC)`.
8. **Dependencies.** Add `torch`, `tokenizers`, `safetensors`, `numpy` to `pyproject.toml`
   (tensorization is CPU-only; no GPU needed). Verify 3.14 wheels at build time; CPU torch is
   sufficient here.
9. **Tests (green on 3.14).** Tokenizer roundtrip + **special-token-ID parity across arms**;
   AXT write/read roundtrip; safetensors loads in torch; **no future-target leakage into
   inputs** (temporal guard); loss/objective-mask correctness; relation-neighborhood
   determinism + no cross-cutoff sampling; **equal-token-budget** enforcement across arms;
   n-ary incidence-layout sanity; AXT traces back to AXC/AXP + tokenizer hashes.

## Guardrails (non-negotiable)

- AXT stays **traceable to AXC/AXP** and **framework-neutral** (torch + JAX read the same
  safetensors).
- **Equal tokens/compute/params across arms** — enforced and recorded here.
- **No future-target leakage** into predictor-visible inputs; temporal cutoffs respected in
  text, relations, and neighborhood sampling.
- **No truth labels** (the upstream `format/validation.py` guard still holds); only
  gauge-invariant geometry slots reserved.
- Everything **ablatable** (side channels, relations, neighborhoods, geometry slot).
- Deterministic + content-hashed. Real components, sane defaults — **no reductions, no
  fine-tuning**; reserve the n-ary and geometry slots rather than removing them.

## Acceptance criteria

From a Plan-1 capsule set (AXC/AXP), `axiom` compiles **AXT bundles for arms A–D** under one
tokenizer + equal token budget; bundles **round-trip** and load as torch tensors via
safetensors; objective masks and the future-summary target are correct with **zero input
leakage**; relation-neighborhood tensors are deterministic and cutoff-respecting; every
bundle traces back to its AXC/AXP + tokenizer hashes; `pytest`, `ruff`, `mypy` green on
Python 3.14.

## References

WP4 (Neural Data Interface) + [../project-knowledge/HoloCapsule_Project_Knowledge.md](../project-knowledge/HoloCapsule_Project_Knowledge.md) §11 (training interface) ·
[../../spec/AXT_TENSOR_BUNDLE.md](../../spec/AXT_TENSOR_BUNDLE.md) · [../../spec/AXF.md](../../spec/AXF.md) · [../concept/TRAINING_BRIDGE.md](../concept/TRAINING_BRIDGE.md) ·
code: `src/hcaps/training_bridge/` (tokenizer, dataset, collator, examples, side_channels, manifests; new: tensorize, neighborhoods), `src/hcaps/format/axt.py` (new), arms in `src/hcaps/falsification/arms.py`.
