from __future__ import annotations

import torch

from hcaps.geometry import GeometryConfig
from hcaps.geometry.batching import ClaimFieldGraphBatch, graph_batch_from_axt_batch
from hcaps.geometry.graph import expand_hyperedges_to_pairwise


def test_empty_graph_batch_works() -> None:
    graph = ClaimFieldGraphBatch.empty()
    assert graph.num_nodes == 0
    assert graph.num_edges == 0
    assert graph.context_edge_index.shape == (2, 0)


def test_graph_batch_from_axt_preserves_stable_structure(
    geometry_axt_batch,
    learned_geometry_config: GeometryConfig,
) -> None:
    graph = graph_batch_from_axt_batch(geometry_axt_batch, learned_geometry_config)
    assert graph.num_nodes == 8
    assert graph.num_edges >= 2
    assert graph.num_context_edges >= 2
    assert graph.node_ids[0].startswith("claim_state:")
    assert graph.temporal_features is not None
    assert graph.lateral_context_features is not None


def test_hyperedge_incidence_expands_without_losing_ids(tiny_loop_graph) -> None:
    assert tiny_loop_graph.hyperedge_incidence is not None
    expansion = expand_hyperedges_to_pairwise(tiny_loop_graph.hyperedge_incidence)
    assert expansion.edge_index.shape[0] == 2
    assert expansion.hyperedge_id.unique().tolist() == [7]
    assert torch.equal(expansion.hyperedge_type_id.unique(), torch.tensor([3]))
