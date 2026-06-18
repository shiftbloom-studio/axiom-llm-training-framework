"""Lateral context transition extraction."""

from __future__ import annotations

from collections.abc import Mapping

from hcaps.geometry.graph import CONTEXT_TRANSITION_TYPE_IDS
from hcaps.model.types import AxiomModelInput


def context_transition_type_id(name: str) -> int:
    return CONTEXT_TRANSITION_TYPE_IDS.get(name, CONTEXT_TRANSITION_TYPE_IDS["unknown_transition"])


def deterministic_context_edges(
    model_input: AxiomModelInput,
    *,
    context_node_indices: list[int],
    provider_node_indices: list[int],
    include_provider: bool,
    include_disagreement: bool,
) -> tuple[list[tuple[int, int]], list[str]]:
    """Build bounded context/provider transitions without using time as context identity."""

    pairs: list[tuple[int, int]] = []
    transition_names: list[str] = []
    batch_size = len(context_node_indices)
    for left in range(batch_size):
        for right in range(batch_size):
            if left == right:
                continue
            transition = _transition_name(model_input, left, right)
            if transition is None:
                continue
            pairs.append((context_node_indices[left], context_node_indices[right]))
            transition_names.append(transition)
    if include_provider:
        for index in range(batch_size):
            pairs.append((context_node_indices[index], provider_node_indices[index]))
            transition_names.append("same_claim_cross_provider")
            pairs.append((provider_node_indices[index], context_node_indices[index]))
            transition_names.append("same_claim_cross_provider")
    if include_provider and include_disagreement and batch_size > 1:
        for index in range(batch_size):
            pairs.append(
                (
                    provider_node_indices[index],
                    provider_node_indices[(index + 1) % batch_size],
                )
            )
            transition_names.append("provider_disagreement_transition")
    return pairs, transition_names


def _transition_name(model_input: AxiomModelInput, left: int, right: int) -> str | None:
    ids = model_input.group("ids")
    claim_left = _int_at(ids, "claim_family_index", left)
    claim_right = _int_at(ids, "claim_family_index", right)
    if claim_left >= 0 and claim_left == claim_right:
        return "same_claim_cross_domain"

    lateral = model_input.group("lateral_context")
    for field_name, transition in (
        ("domain_idx", "same_claim_cross_domain"),
        ("method_context_idx", "same_claim_cross_method"),
        ("community_idx", "same_claim_cross_community"),
        ("language_idx", "same_claim_cross_language"),
        ("source_context_idx", "same_claim_cross_source_type"),
    ):
        left_value = _int_at(lateral, field_name, left)
        right_value = _int_at(lateral, field_name, right)
        if left_value >= 0 and right_value >= 0 and left_value != right_value:
            return transition
    return None


def _int_at(group: Mapping[str, object], field_name: str, index: int) -> int:
    value = group.get(field_name)
    if value is None or not hasattr(value, "detach"):
        return -1
    tensor = value.detach()
    if tensor.ndim == 0:
        return int(tensor.item()) if index == 0 else -1
    if tensor.ndim == 1:
        return int(tensor[index].item()) if index < tensor.shape[0] else -1
    reshaped = tensor.reshape(tensor.shape[0], -1)
    return int(reshaped[index, 0].item()) if index < reshaped.shape[0] else -1
