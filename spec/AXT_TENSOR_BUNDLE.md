# AXT v0.1 Tensor Bundle

AXT is the future compiled training tensor format for AXF datasets.

AXT is specified in v0.1 but not implemented in this step. Tensor compilation
belongs to the training bridge stage, after tokenizer, collator, side-channel,
and objective-mask contracts are stable.

## Intended Contents

An AXT bundle may contain:

- token IDs and attention masks;
- loss masks;
- capsule IDs and claim-family IDs;
- relation tensors;
- provenance tensors;
- epistemic scalar tensors;
- context identifiers or context embeddings;
- objective masks;
- train/validation/test split metadata;
- source AXF/AXP manifest hashes.

## Non-Goals For v0.1

AXT v0.1 does not define:

- model architecture;
- tokenizer choice;
- training loop;
- geometry learning;
- learned connection matrices;
- benchmark claims.

AXT must remain traceable back to AXC records and AXP manifests.
