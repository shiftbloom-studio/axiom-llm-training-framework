"""Deterministic provider merge policy."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hcaps.extraction.contracts import ProviderDisagreement
from hcaps.providers.base import ProviderResponse, hash_normalized_output


class MergeResult(BaseModel):
    """Merged provider response plus preserved disagreement metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response: ProviderResponse
    disagreements: list[ProviderDisagreement] = Field(default_factory=list)
    merge_policy: str


def merge_provider_outputs(
    primary: ProviderResponse,
    escalation: ProviderResponse | None,
    *,
    merge_policy: str,
) -> MergeResult:
    """Merge provider outputs deterministically without hiding disagreement."""

    if escalation is None:
        response = primary.model_copy(
            update={
                "provider_trace": primary.provider_trace.model_copy(
                    update={"merge_decision": "single_provider"}
                )
            }
        )
        return MergeResult(response=response, merge_policy=merge_policy)

    primary_hash = hash_normalized_output(primary.normalized_output)
    escalation_hash = hash_normalized_output(escalation.normalized_output)
    if primary_hash == escalation_hash:
        response = primary.model_copy(
            update={
                "provider_trace": primary.provider_trace.model_copy(
                    update={"merge_decision": "provider_outputs_equivalent"}
                )
            }
        )
        return MergeResult(response=response, merge_policy=merge_policy)

    selected = _select_response(primary, escalation)
    rejected = escalation if selected is primary else primary
    selected_label = "primary" if selected is primary else "escalation"
    selected_keys = sorted(selected.normalized_output)
    rejected_keys = sorted(set(rejected.normalized_output) - set(selected.normalized_output))
    response = selected.model_copy(
        update={
            "provider_trace": selected.provider_trace.model_copy(
                update={"merge_decision": f"selected_{selected_label}"}
            )
        }
    )
    disagreement = ProviderDisagreement(
        local_output_hash=primary_hash,
        remote_output_hash=escalation_hash,
        disagreement_type="normalized_output_hash_mismatch",
        merge_policy=merge_policy,
        merged_output_hash=hash_normalized_output(response.normalized_output),
        selected_fields=selected_keys,
        rejected_fields=rejected_keys,
        human_review_required=True,
    )
    return MergeResult(
        response=response,
        disagreements=[disagreement],
        merge_policy=merge_policy,
    )


def _select_response(primary: ProviderResponse, escalation: ProviderResponse) -> ProviderResponse:
    primary_confidence = primary.confidence or 0.0
    escalation_confidence = escalation.confidence or 0.0
    if escalation_confidence > primary_confidence:
        return escalation
    if primary_confidence > escalation_confidence:
        return primary
    if len(escalation.normalized_output) > len(primary.normalized_output):
        return escalation
    return primary
