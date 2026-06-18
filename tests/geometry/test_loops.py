from __future__ import annotations

from hcaps.geometry.batching import ClaimFieldGraphBatch
from hcaps.geometry.loops import sample_loops


def test_loop_sampler_is_deterministic_and_bounded(
    tiny_loop_graph,
    learned_geometry_config,
) -> None:
    first = sample_loops(tiny_loop_graph, learned_geometry_config)
    second = sample_loops(tiny_loop_graph, learned_geometry_config)
    assert first.diagnostics == second.diagnostics
    assert first.loop_path_index.tolist() == second.loop_path_index.tolist()
    assert first.loop_path_index.shape[0] <= learned_geometry_config.max_loops_per_batch


def test_no_loops_returns_masked_empty_result(learned_geometry_config) -> None:
    graph = ClaimFieldGraphBatch.empty()
    result = sample_loops(graph, learned_geometry_config)
    assert result.diagnostics["no_loops"] is True
    assert result.loop_mask.numel() == 0
