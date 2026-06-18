# Evaluation Protocol

Evaluation is implemented in later steps. This document records the comparison
arms that Step 1 must keep possible.

## Future Comparison Arms

1. Flat text baseline.
2. Structured text baseline.
3. HoloCapsule text-only.
4. HoloCapsule with side channels.
5. Context-shuffled ablation.
6. No-provenance ablation.
7. No-relations ablation.
8. Popularity, degree, and recency controls.

## Evaluation Dimensions

Future evaluation should include:

- next-token or validation loss under equal token budget;
- relation prediction;
- support and contradiction classification;
- provenance recovery;
- uncertainty calibration;
- temporal stability prediction;
- outdated-belief handling;
- source sensitivity;
- contamination and target-leakage checks.

## Falsification Checks

If structured text beats capsule variants and side channels add nothing, the
project should prioritize text view quality over capsule architecture.

If context shuffling does not hurt geometric/HKR variants, the geometry branch is
probably not carrying real context signal.

If popularity or degree controls explain the gains, redundancy and provenance
modeling must be redesigned.

If no capsule variant beats flat text under equal conditions, the core
claim-field hypothesis should be narrowed, redesigned, or rejected.
