# Training Bridge

The training bridge is the component that turns claim-centric AXF
capsules into model-ready training examples. It sits between the
structured data contract defined in earlier milestones and the
machine learning models that consume token sequences and optional side
channels.

## Modes

The bridge supports three text rendering modes:

* **flat_text** – extracts only the canonical claim or best surface
  representation. Use this as a baseline comparable to plain text
  corpora.
* **structured_text** – includes high-level labels such as the claim,
  context and sources. This mode exposes epistemic structure without
  overwhelming detail.
* **capsule** – outputs a verbose, labelled representation of the
  entire capsule. It is useful for models that can leverage rich
  structural cues.

## Tokenization

Tokenization is performed by objects implementing the
`TokenizerProtocol`. A simple baseline tokenizer splits on whitespace
and lowercases tokens. Users may substitute a BPE or WordPiece
tokenizer by implementing the protocol and ensuring that the same
special token IDs are exposed.

## Side Channels

Side-channel features are numeric values derived from the capsule’s
epistemic state, relation counts, provenance counts and geometry
flags. They are provided separately from the tokenized text. Models
may optionally consume these features to improve calibration or
reasoning without encoding epistemic information in the language
stream itself.

## Data Loader and Collator

The `AxcTrainingDataset` reads AXC capsules (or canonical capsule
streams inside AXP packages) into memory and yields tokenized
examples. The `TrainingCollator` pads variable-length sequences to
uniform length within a batch, produces attention masks and label
masks and stacks side-channel vectors.

## Manifest

A manifest records the configuration used to produce a dataset: input
paths, tokenizer metadata, rendering mode, sequence length limits,
whether side channels were included and the number of records. Use
manifests to track experiments and ensure reproducibility.
