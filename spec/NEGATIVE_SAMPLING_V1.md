# AXF v1 Negative Sampling Metadata

P1 negative pools become P3 AXT negative sample tensors. P2 defines metadata
only; it does not implement training losses.

## Required Metadata

Each negative item must include:

- `negative_id`
- `negative_type`
- `source_id`
- `source_claim_family_id`
- `target_id`
- `target_claim_family_id`
- `hardness_score`
- `sampling_method`
- `sampling_basis`
- `sampling_seed`
- optional `provider_disagreement_ref`
- `temporal_validity`
- `split_eligibility`

## Negative Types

- `near_topic_negative`
- `near_claim_negative`
- `same_context_unrelated`
- `same_source_unrelated`
- `contradiction_candidate`
- `supersession_candidate`
- `temporal_negative`
- `provider_disagreement_negative`
- `hard_relation_negative`
- `provenance_negative`

## Loss Eligibility

| negative_type | Eligible objective |
|---|---|
| near_topic_negative | context discrimination loss |
| near_claim_negative | relation contrastive loss |
| same_context_unrelated | context discrimination loss |
| same_source_unrelated | provenance recovery loss |
| contradiction_candidate | relation contrastive loss |
| supersession_candidate | temporal ordering loss |
| temporal_negative | temporal ordering loss |
| provider_disagreement_negative | provider-disagreement auxiliary loss |
| hard_relation_negative | relation contrastive loss |
| provenance_negative | provenance recovery loss |

## Reproducibility

Negative sampling must be deterministic, seeded, manifestable, and ablatable.
Provider disagreement negatives must preserve the disagreement reference and
must not be flattened away as ordinary labels.
