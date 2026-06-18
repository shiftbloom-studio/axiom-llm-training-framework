"""Geometry control utilities for P6 ablation arms."""

from __future__ import annotations

from dataclasses import replace

import torch

from hcaps.geometry.batching import ClaimFieldGraphBatch


def shuffle_context_edges(graph_batch: ClaimFieldGraphBatch, *, seed: int) -> ClaimFieldGraphBatch:
    if graph_batch.context_edge_index.shape[1] == 0:
        return graph_batch
    generator = torch.Generator(device=graph_batch.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(graph_batch.context_edge_index.shape[1], generator=generator)
    shuffled_targets = graph_batch.context_edge_index[1].index_select(0, permutation)
    shuffled = torch.stack([graph_batch.context_edge_index[0], shuffled_targets], dim=0)
    return _replace_context_edges(graph_batch, shuffled)


def shuffle_provider_features(
    graph_batch: ClaimFieldGraphBatch,
    *,
    seed: int,
) -> ClaimFieldGraphBatch:
    if graph_batch.provider_features is None or graph_batch.provider_features.shape[0] == 0:
        return graph_batch
    generator = torch.Generator(device=graph_batch.provider_features.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(graph_batch.provider_features.shape[0], generator=generator)
    return replace(
        graph_batch,
        provider_features=graph_batch.provider_features.index_select(0, permutation),
    )


def degree_preserving_rewire(
    graph_batch: ClaimFieldGraphBatch,
    *,
    seed: int,
) -> ClaimFieldGraphBatch:
    """Rewire context targets while preserving source degrees and target multiset."""

    return shuffle_context_edges(graph_batch, seed=seed)


def _replace_context_edges(
    graph_batch: ClaimFieldGraphBatch,
    context_edge_index: torch.Tensor,
) -> ClaimFieldGraphBatch:
    return replace(graph_batch, context_edge_index=context_edge_index)
