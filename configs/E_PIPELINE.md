# Arm E pipeline — canonical configs

The project's first-class path is **arm E**: structured-native training with in-model,
while-training learned geometry (the HKR hypothesis). Use only these for it:

- training: `configs/training/pioneer_e_structured_native_geometry.yaml`
- model:    `configs/model/structured_native_research_geometry.yaml` (geometry_provider_injected)
- geometry: `configs/geometry/geometry_learned_pioneer.yaml`

Stages: `axiom harvest` → `axiom prepare` → `axiom train` (or `axiom full`).

Everything else under `configs/` (the smoke variants, `experiments/`, `scoring/`,
`verdict/`, `evaluation/`) belongs to the research / falsification arms (A–J). They are
kept for later but are **not** part of the E pipeline; their separate command surface is
`axiom experiment` / `score` / `verdict` / `falsify`.
