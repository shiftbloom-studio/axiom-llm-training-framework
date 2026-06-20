# Axiom v0.1.0-alpha — research checkpoint (untested)

Pre-release checkpoint marking the point where the structured-native + learned
geometry hypothesis (HKR) is wired end-to-end. This is a milestone, **not** a
validated release: it has not been run or benchmarked, and may produce a strong
model or a useless one — that is the open question by design.

## What this checkpoint locks in
- **Learned geometry actually runs on standalone runs.** Added an injected-mode
  model config (`structured_native_research_geometry`) and a guard in
  `AxiomTrainer` that *raises* when geometry is requested but the model config
  would bypass the learned provider — preventing arm E from silently degrading
  to arm D.
- **Geometry regularizer weights are wired.** The geometry YAML now controls each
  term; the terms that flatten the geometry (`transport_identity_bias`,
  `curvature_energy`) are set to zero in the pioneer config.
- **Numerically-safe geometry.** The geometry linear algebra (`matrix_exp` /
  parallel transport) is computed in a float32 island, so bf16 can be enabled
  later for speed without destabilizing the transport.
- **Standalone arm-E pioneer configs** (training + geometry) for a single,
  data-driven run that is bounded by passes over the data, not wall-clock time.

## Status / caveats
- Not run, not benchmarked.
- `pytest` / `ruff` / `mypy` not executed for this checkpoint — run locally before relying on it.
- Requires Python 3.14 and `torch >= 2.12.1` (cp314 wheels exist; GPU on 3.14 may need a nightly build).
