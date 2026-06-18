# AXF v1 Loss And Target Mask Semantics

Not every capsule has every target. Missing target availability must never be
interpreted as a negative target.

## Required Loss Masks

- `loss_mask_text_projection`
- `loss_mask_relation_prediction`
- `loss_mask_provenance_recovery`
- `loss_mask_stability_prediction`
- `loss_mask_uncertainty_calibration`
- `loss_mask_future_summary`
- `loss_mask_temporal_prediction`
- `loss_mask_geometry_observables`
- `loss_mask_provider_context`
- `loss_mask_negative_sampling`

## Required Availability Masks

- `available_relation_targets`
- `available_provenance_targets`
- `available_epistemic_targets`
- `available_future_targets`
- `available_geometry_targets`
- `available_text_targets`
- `available_evaluation_references`

## Target State Vocabulary

P3 must distinguish:

- `unknown`
- `missing`
- `not_applicable`
- `not_predictor_visible`
- `target_only`
- `masked_by_split`
- `available`

## Mask Rules

- `unknown`: no claim about availability; loss mask is false.
- `missing`: expected but absent; loss mask is false and quality warning may be
  recorded.
- `not_applicable`: objective does not apply; loss mask is false.
- `not_predictor_visible`: field may not enter predictor tensors or text.
- `target_only`: field may enter target tensors only.
- `masked_by_split`: target exists but is excluded for the current split.
- `available`: target may participate in its declared objective.

## Temporal And Evaluation Guardrails

Future-facing fields, evaluation references, gold references, answer keys, and
target-only material must not enter predictor text or predictor-side AXT tensors.

## Relationship To AXT

AXT must store both availability and loss masks. Availability describes whether
a target exists. Loss masks describe whether a training objective may consume it
in a particular arm/split/curriculum step.
