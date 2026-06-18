"""Claim-field graph schemas and construction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

NODE_TYPE_IDS: dict[str, int] = {
    "claim_state": 0,
    "claim_family": 1,
    "context": 2,
    "provider": 3,
    "source": 4,
    "community": 5,
    "method": 6,
}

CONTEXT_TRANSITION_TYPE_IDS: dict[str, int] = {
    "same_claim_cross_domain": 0,
    "same_claim_cross_provider": 1,
    "same_claim_cross_source_type": 2,
    "same_claim_cross_method": 3,
    "same_claim_cross_community": 4,
    "same_claim_cross_language": 5,
    "same_claim_cross_benchmark_family": 6,
    "provider_disagreement_transition": 7,
    "manual_reference_transition": 8,
    "unknown_transition": 9,
}

MIN_HYPEREDGE_ARITY = 2


@dataclass(frozen=True)
class GeometryMasks:
    node_mask: Tensor
    relation_edge_mask: Tensor
    context_edge_mask: Tensor
    path_mask: Tensor
    loop_mask: Tensor


@dataclass(frozen=True)
class HyperedgeIncidence:
    hyperedge_id: Tensor
    node_id: Tensor
    role_id: Tensor
    hyperedge_type_id: Tensor

    @property
    def device(self) -> torch.device:
        return self.node_id.device


@dataclass(frozen=True)
class PairwiseExpansion:
    edge_index: Tensor
    hyperedge_id: Tensor
    role_pair: Tensor
    hyperedge_type_id: Tensor


def empty_geometry_masks(*, device: torch.device) -> GeometryMasks:
    return GeometryMasks(
        node_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        relation_edge_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        context_edge_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        path_mask=torch.zeros((0, 0), dtype=torch.bool, device=device),
        loop_mask=torch.zeros((0, 0), dtype=torch.bool, device=device),
    )


def expand_hyperedges_to_pairwise(incidence: HyperedgeIncidence) -> PairwiseExpansion:
    """Expand incidence rows to pairwise edges while preserving ids and roles."""

    rows: list[tuple[int, int, int, int, int, int]] = []
    if incidence.node_id.numel() == 0:
        return PairwiseExpansion(
            edge_index=torch.zeros((2, 0), dtype=torch.long, device=incidence.device),
            hyperedge_id=torch.zeros((0,), dtype=torch.long, device=incidence.device),
            role_pair=torch.zeros((0, 2), dtype=torch.long, device=incidence.device),
            hyperedge_type_id=torch.zeros((0,), dtype=torch.long, device=incidence.device),
        )
    for hyperedge in torch.unique(incidence.hyperedge_id).detach().cpu().tolist():
        mask = incidence.hyperedge_id.eq(int(hyperedge))
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() < MIN_HYPEREDGE_ARITY:
            continue
        type_id = int(incidence.hyperedge_type_id[indices[0]].item())
        for left_pos in range(indices.numel()):
            for right_pos in range(left_pos + 1, indices.numel()):
                left_index = int(indices[left_pos].item())
                right_index = int(indices[right_pos].item())
                rows.append(
                    (
                        int(incidence.node_id[left_index].item()),
                        int(incidence.node_id[right_index].item()),
                        int(hyperedge),
                        int(incidence.role_id[left_index].item()),
                        int(incidence.role_id[right_index].item()),
                        type_id,
                    )
                )
                rows.append(
                    (
                        int(incidence.node_id[right_index].item()),
                        int(incidence.node_id[left_index].item()),
                        int(hyperedge),
                        int(incidence.role_id[right_index].item()),
                        int(incidence.role_id[left_index].item()),
                        type_id,
                    )
                )
    if not rows:
        return PairwiseExpansion(
            edge_index=torch.zeros((2, 0), dtype=torch.long, device=incidence.device),
            hyperedge_id=torch.zeros((0,), dtype=torch.long, device=incidence.device),
            role_pair=torch.zeros((0, 2), dtype=torch.long, device=incidence.device),
            hyperedge_type_id=torch.zeros((0,), dtype=torch.long, device=incidence.device),
        )
    edge_index = torch.tensor([[row[0], row[1]] for row in rows], dtype=torch.long).t()
    return PairwiseExpansion(
        edge_index=edge_index.to(incidence.device),
        hyperedge_id=torch.tensor(
            [row[2] for row in rows],
            dtype=torch.long,
            device=incidence.device,
        ),
        role_pair=torch.tensor(
            [[row[3], row[4]] for row in rows],
            dtype=torch.long,
            device=incidence.device,
        ),
        hyperedge_type_id=torch.tensor(
            [row[5] for row in rows],
            dtype=torch.long,
            device=incidence.device,
        ),
    )


def edge_index_from_pairs(pairs: list[tuple[int, int]], *, device: torch.device) -> Tensor:
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
