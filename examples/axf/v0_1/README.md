# AXF v0.1 Conformance Fixtures

This directory contains synthetic, canonical AXF v0.1 fixtures. They are executable
format-contract examples, not benchmark data and not model-training data.

AXF is the Axiom Exchange Format family:

- AXC is the Axiom Capsule Stream: line-oriented claim-state capsules.
- AXP is the Axiom Package: a dataset package with manifests, hashes, splits, and reports.
- AXT is the Axiom Tensor Bundle: the future compiled tensor format, specified but not
  implemented in this step.

Valid fixtures:

- `minimal_capsules.axc` contains one minimal claim-state capsule.
- `full_capsules.axc` contains two capsules, provenance spans, epistemic measures,
  disabled geometry, and `supports` / `supersedes` relation examples.
- `minimal_dataset.axp/` is a minimal package with AXC data, source metadata, relation
  metadata, splits, file manifests, hash manifests, construction report, and leakage report.
- `minimal_tensor_bundle.axt` records the planned AXT manifest surface without tokenizer outputs
  or tensors.

AXC files use the public `.axc` suffix even though v0.1 is internally UTF-8
newline-delimited canonical JSON. The suffix names the format contract; NDJSON
is the current encoding strategy.

Intentionally invalid fixtures:

- `invalid/future_leakage_capsule.axc` uses predictor-visible source dates later
  than `valid_as_of`.
- `invalid/missing_provenance_capsule.axc` references a source span whose source is
  absent from provenance.
- `invalid/forbidden_truth_label_capsule.axc` includes a dogmatic truth-style field.
- `invalid/unsupported_version_capsule.axc` declares an unsupported AXC version.

Run the conformance checks with:

```bash
pytest tests/format
```

These fixtures are deliberately tiny so that a maintainer can inspect every byte. They use
neutral synthetic scientific-style claims and do not assert real-world benchmark results.
