# ML/Software Benchmark Corpus Fixture

This fixture is synthetic and license-safe. It exercises P1 corpus ingress with
deterministic provider extraction, provider cache records, source registry
artifacts, claim families, negative pools, gold-reference candidates, AXC, and
AXP output.

Run:

```bash
axiom corpus build --config examples/corpus/ml_software_benchmarks/configs/deterministic_only.yaml
```

The checked-in source is intentionally tiny. Real ML/software benchmark corpora
should be generated locally into `artifacts/` or another ignored artifact root.
