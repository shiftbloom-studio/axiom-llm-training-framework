"""Developer-facing geometry command helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import torch

from hcaps.axt import AxtBatchCollator, AxtDataset
from hcaps.geometry.batching import ClaimFieldGraphBatch, graph_batch_from_axt_batch
from hcaps.geometry.config import GeometryConfig, GeometryMode, config_summary
from hcaps.geometry.export import geometry_observables_to_axc_out_fields
from hcaps.geometry.loops import sample_loops
from hcaps.geometry.module import LearnedGeometryModule, NoGeometryModule, NonGeometricContextMixer
from hcaps.geometry.reference import observables_reference


def inspect_axt_geometry(bundle_path: Path, *, batch_size: int = 4) -> dict[str, Any]:
    graph = _load_graph(bundle_path, GeometryConfig(mode=GeometryMode.OFF), batch_size=batch_size)
    return {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "num_context_edges": graph.num_context_edges,
        "num_paths": int(graph.path_edge_index.shape[0]),
        "num_loops": int(graph.loop_path_index.shape[0]),
        "has_hyperedge_incidence": graph.hyperedge_incidence is not None,
        "context_node_ids": list(graph.context_node_ids),
    }


def sample_axt_loops(
    bundle_path: Path,
    *,
    output: Path,
    config: GeometryConfig,
    batch_size: int = 4,
) -> dict[str, Any]:
    graph = _load_graph(bundle_path, config, batch_size=batch_size)
    loop_sample = sample_loops(graph, config)
    payload = {
        "loop_path_index": loop_sample.loop_path_index.detach().cpu().tolist(),
        "loop_mask": loop_sample.loop_mask.detach().cpu().tolist(),
        "diagnostics": loop_sample.diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return payload


def compute_reference_geometry(
    bundle_path: Path,
    *,
    output: Path,
    config: GeometryConfig,
    batch_size: int = 4,
) -> dict[str, Any]:
    graph = _load_graph(bundle_path, config, batch_size=batch_size)
    del graph
    identity_observables = observables_reference(torch.eye(config.fiber_dim).numpy())
    payload = {
        "mode": "reference",
        "observables": identity_observables,
        "note": "off-loop numpy reference identity holonomy for smoke sanity",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return payload


def smoke_geometry(
    bundle_path: Path,
    *,
    config: GeometryConfig,
    batch_size: int = 4,
) -> dict[str, Any]:
    graph = _load_graph(bundle_path, config, batch_size=batch_size)
    node_states = _deterministic_node_states(
        graph.num_nodes,
        config.geometry_feature_dim,
        seed=config.seed,
    )
    if config.mode == GeometryMode.OFF:
        output = NoGeometryModule(config)(node_states=node_states, graph_batch=graph)
    elif config.parameter_matched:
        output = NonGeometricContextMixer(config)(node_states=node_states, graph_batch=graph)
    else:
        output = LearnedGeometryModule(config)(node_states=node_states, graph_batch=graph)
    return {
        "config": config_summary(config),
        "conditioning_shape": tuple(output.conditioning_features.shape),
        "regularizer_terms": {
            key: float(value.detach().cpu().item())
            for key, value in output.regularizer_terms.items()
        },
        "diagnostics": output.diagnostics.to_dict(),
        "axc_out_fields": geometry_observables_to_axc_out_fields(
            mode=output.mode,
            enabled=output.mode != GeometryMode.OFF,
            observables=output.observables,
        ),
    }


def _load_graph(
    bundle_path: Path,
    config: GeometryConfig,
    *,
    batch_size: int,
) -> ClaimFieldGraphBatch:
    dataset = AxtDataset(bundle_path)
    if len(dataset) == 0:
        raise ValueError("AXT bundle has no selected records")
    records = [dataset[index] for index in range(min(batch_size, len(dataset)))]
    batch = AxtBatchCollator()(records)
    return graph_batch_from_axt_batch(batch, config)


def _deterministic_node_states(count: int, dim: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn((count, dim), generator=generator) * 0.01
