from __future__ import annotations

from hcaps.geometry.config import GeometryConfig, GeometryMode
from hcaps.geometry.module import NoGeometryModule


def test_geometry_off_config_mode_is_explicit(tiny_loop_graph) -> None:
    config = GeometryConfig(mode=GeometryMode.OFF, geometry_feature_dim=8)
    output = NoGeometryModule(config, input_dim=8)(
        node_states=tiny_loop_graph.node_type_ids.float().reshape(-1, 1).expand(-1, 8),
        graph_batch=tiny_loop_graph,
    )
    assert output.mode == GeometryMode.OFF
    assert output.diagnostics.values["geometry_mode"] == "off"
