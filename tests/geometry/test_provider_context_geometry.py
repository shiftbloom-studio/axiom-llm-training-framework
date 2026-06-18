from __future__ import annotations

import torch

from hcaps.geometry.batching import graph_batch_from_axt_batch
from hcaps.geometry.controls import shuffle_context_edges, shuffle_provider_features


def test_provider_context_can_create_transitions(
    geometry_axt_batch,
    learned_geometry_config,
) -> None:
    graph = graph_batch_from_axt_batch(geometry_axt_batch, learned_geometry_config)
    assert graph.provider_features is not None
    assert graph.num_context_edges > 0


def test_provider_shuffle_is_deterministic(geometry_axt_batch, learned_geometry_config) -> None:
    graph = graph_batch_from_axt_batch(geometry_axt_batch, learned_geometry_config)
    first = shuffle_provider_features(graph, seed=42)
    second = shuffle_provider_features(graph, seed=42)
    assert first.provider_features is not None
    assert second.provider_features is not None
    assert torch.equal(first.provider_features, second.provider_features)


def test_context_shuffle_is_deterministic(geometry_axt_batch, learned_geometry_config) -> None:
    graph = graph_batch_from_axt_batch(geometry_axt_batch, learned_geometry_config)
    first = shuffle_context_edges(graph, seed=7)
    second = shuffle_context_edges(graph, seed=7)
    assert torch.equal(first.context_edge_index, second.context_edge_index)
