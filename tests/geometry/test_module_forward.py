from __future__ import annotations

import torch

from hcaps.geometry.module import LearnedGeometryModule, NoGeometryModule, NonGeometricContextMixer


def test_learned_module_returns_conditioning_regularizers_and_gradients(
    tiny_loop_graph,
    learned_geometry_config,
) -> None:
    module = LearnedGeometryModule(learned_geometry_config, input_dim=64)
    node_states = torch.randn((tiny_loop_graph.num_nodes, 64), requires_grad=True)
    output = module(node_states=node_states, graph_batch=tiny_loop_graph)
    assert output.conditioning_features.shape == (tiny_loop_graph.num_nodes, 64)
    assert "connection_norm" in output.regularizer_terms
    loss = output.conditioning_features.sum() + output.regularizer_terms["connection_norm"]
    loss.backward()
    assert node_states.grad is not None
    assert torch.isfinite(node_states.grad).all()


def test_geometry_off_returns_zeros(tiny_loop_graph, learned_geometry_config) -> None:
    node_states = torch.randn((tiny_loop_graph.num_nodes, 64))
    output = NoGeometryModule(learned_geometry_config, input_dim=64)(
        node_states=node_states,
        graph_batch=tiny_loop_graph,
    )
    assert torch.count_nonzero(output.conditioning_features) == 0
    assert output.axc_out_fields["geometry"]["enabled"] is False


def test_non_geometric_baseline_is_shape_compatible(
    tiny_loop_graph,
    learned_geometry_config,
) -> None:
    node_states = torch.randn((tiny_loop_graph.num_nodes, 64))
    output = NonGeometricContextMixer(learned_geometry_config, input_dim=64)(
        node_states=node_states,
        graph_batch=tiny_loop_graph,
    )
    assert output.conditioning_features.shape == (tiny_loop_graph.num_nodes, 64)
    assert output.diagnostics.values["raw_gauge_matrices_emitted"] is False
