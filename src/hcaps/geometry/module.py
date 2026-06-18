"""Main torch-native learned geometry modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from hcaps.geometry.batching import ClaimFieldGraphBatch, graph_batch_from_model_input
from hcaps.geometry.config import GeometryConfig, GeometryMode
from hcaps.geometry.connection import LearnedConnection
from hcaps.geometry.diagnostics import GeometryDiagnostics, build_geometry_diagnostics
from hcaps.geometry.export import geometry_observables_to_axc_out_fields
from hcaps.geometry.loops import sample_loops
from hcaps.geometry.observables import (
    GeometryObservables,
    compute_observables,
    empty_observables,
)
from hcaps.geometry.regularizers import geometry_regularizers, zero_regularizers
from hcaps.geometry.transport import compose_path_transport, path_consistency
from hcaps.model.geometry_hooks import GeometryContext
from hcaps.model.types import AxiomModelInput


@dataclass(frozen=True)
class GeometryOutput:
    mode: GeometryMode
    conditioning_features: Tensor
    observables: GeometryObservables
    regularizer_terms: dict[str, Tensor]
    diagnostics: GeometryDiagnostics
    axc_out_fields: dict[str, Any]


class LearnedGeometryModule(nn.Module):
    """Learn context-transport geometry over claim-field graph batches."""

    def __init__(self, config: GeometryConfig, *, input_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.input_dim = input_dim or config.geometry_feature_dim
        self.connection = LearnedConnection(config)
        self.fiber_projection = nn.Linear(self.input_dim, config.fiber_dim)
        self.conditioning_projection = nn.Sequential(
            nn.Linear(config.fiber_dim + 4, config.geometry_feature_dim),
            nn.LayerNorm(config.geometry_feature_dim),
            nn.GELU(),
        )

    def forward(
        self,
        *,
        node_states: Tensor,
        graph_batch: ClaimFieldGraphBatch,
        context_features: Tensor | None = None,
        temporal_features: Tensor | None = None,
        masks: object | None = None,
    ) -> GeometryOutput:
        del context_features, temporal_features, masks
        if self.config.mode == GeometryMode.OFF:
            return NoGeometryModule(self.config, input_dim=node_states.shape[-1]).forward(
                node_states=node_states,
                graph_batch=graph_batch,
            )
        if graph_batch.num_nodes == 0:
            return _empty_output(
                self.config,
                node_states=node_states,
                graph_batch=graph_batch,
                mode=self.config.mode,
            )

        graph_batch = _ensure_loops(graph_batch, self.config)
        fiber_state = self.fiber_projection(node_states)
        relation_type_ids = _context_relation_ids(graph_batch)
        connection = self.connection(
            graph_batch.context_transition_type_ids,
            relation_type_ids=relation_type_ids,
        )
        path_transports = compose_path_transport(
            connection.transport_matrices,
            graph_batch.path_edge_index,
            graph_batch.path_mask,
            fiber_dim=self.config.fiber_dim,
        )
        loop_transports = compose_path_transport(
            connection.transport_matrices,
            graph_batch.loop_path_index,
            graph_batch.loop_mask,
            fiber_dim=self.config.fiber_dim,
        )
        path_consistency_value = path_consistency(path_transports)
        context_lability_value = _context_lability(fiber_state, graph_batch)
        observables = compute_observables(
            loop_transports,
            graph_batch.loop_mask,
            path_consistency_value=path_consistency_value,
            context_lability_value=context_lability_value,
        )
        conditioning_features = self._conditioning_features(
            fiber_state,
            graph_batch,
            observables,
        )
        regularizers = geometry_regularizers(
            connection_matrices=connection.connection_matrices,
            transport_matrices=connection.transport_matrices,
            path_consistency_value=path_consistency_value,
            conditioning_features=conditioning_features,
            observables=observables,
        )
        diagnostics = build_geometry_diagnostics(
            mode=self.config.mode,
            graph_batch=graph_batch,
            connection=connection,
            observables=observables,
            no_loops=graph_batch.loop_path_index.shape[0] == 0,
            empty_graph=False,
        )
        return GeometryOutput(
            mode=self.config.mode,
            conditioning_features=conditioning_features,
            observables=observables,
            regularizer_terms=regularizers,
            diagnostics=diagnostics,
            axc_out_fields=geometry_observables_to_axc_out_fields(
                mode=self.config.mode,
                enabled=True,
                observables=observables,
            ),
        )

    def _conditioning_features(
        self,
        fiber_state: Tensor,
        graph_batch: ClaimFieldGraphBatch,
        observables: GeometryObservables,
    ) -> Tensor:
        node_degree = _node_degree_features(graph_batch, dtype=fiber_state.dtype)
        global_features = torch.stack(
            [
                observables.curvature_score,
                observables.path_consistency_score,
                observables.context_lability_score,
                observables.loop_count.to(dtype=fiber_state.dtype).clamp(max=128.0) / 128.0,
            ]
        )
        global_features = global_features.reshape(1, 4).expand(fiber_state.shape[0], -1)
        del node_degree
        return cast(
            Tensor,
            self.conditioning_projection(torch.cat([fiber_state, global_features], dim=-1)),
        )


class NoGeometryModule(nn.Module):
    """Geometry-off ablation returning neutral features and empty observables."""

    def __init__(self, config: GeometryConfig, *, input_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.input_dim = input_dim or config.geometry_feature_dim

    def forward(
        self,
        *,
        node_states: Tensor,
        graph_batch: ClaimFieldGraphBatch,
        context_features: Tensor | None = None,
        temporal_features: Tensor | None = None,
        masks: object | None = None,
    ) -> GeometryOutput:
        del context_features, temporal_features, masks
        conditioning = torch.zeros(
            (node_states.shape[0], self.config.geometry_feature_dim),
            dtype=node_states.dtype,
            device=node_states.device,
        )
        observables = empty_observables(device=node_states.device, dtype=node_states.dtype)
        return GeometryOutput(
            mode=GeometryMode.OFF,
            conditioning_features=conditioning,
            observables=observables,
            regularizer_terms=zero_regularizers(device=node_states.device, dtype=node_states.dtype),
            diagnostics=build_geometry_diagnostics(
                mode=GeometryMode.OFF,
                graph_batch=graph_batch,
                connection=None,
                observables=observables,
                no_loops=True,
                empty_graph=graph_batch.num_nodes == 0,
            ),
            axc_out_fields=geometry_observables_to_axc_out_fields(
                mode=GeometryMode.OFF,
                enabled=False,
                observables=observables,
            ),
        )


class NonGeometricContextMixer(nn.Module):
    """Parameter-bearing non-geometric control without transport/holonomy."""

    def __init__(self, config: GeometryConfig, *, input_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.input_dim = input_dim or config.geometry_feature_dim
        self.mixer = nn.Sequential(
            nn.Linear(self.input_dim + 4, config.geometry_feature_dim),
            nn.GELU(),
            nn.Linear(config.geometry_feature_dim, config.geometry_feature_dim),
            nn.LayerNorm(config.geometry_feature_dim),
        )

    def forward(
        self,
        *,
        node_states: Tensor,
        graph_batch: ClaimFieldGraphBatch,
        context_features: Tensor | None = None,
        temporal_features: Tensor | None = None,
        masks: object | None = None,
    ) -> GeometryOutput:
        del context_features, temporal_features, masks
        observables = empty_observables(device=node_states.device, dtype=node_states.dtype)
        degree_features = _node_degree_features(graph_batch, dtype=node_states.dtype)
        if degree_features.shape[0] != node_states.shape[0]:
            degree_features = torch.zeros(
                (node_states.shape[0], 4),
                dtype=node_states.dtype,
                device=node_states.device,
            )
        conditioning = cast(Tensor, self.mixer(torch.cat([node_states, degree_features], dim=-1)))
        return GeometryOutput(
            mode=GeometryMode.OFF,
            conditioning_features=conditioning,
            observables=observables,
            regularizer_terms=zero_regularizers(device=node_states.device, dtype=node_states.dtype),
            diagnostics=build_geometry_diagnostics(
                mode=GeometryMode.OFF,
                graph_batch=graph_batch,
                connection=None,
                observables=observables,
                no_loops=True,
                empty_graph=graph_batch.num_nodes == 0,
            ),
            axc_out_fields=geometry_observables_to_axc_out_fields(
                mode=GeometryMode.OFF,
                enabled=False,
                observables=observables,
            ),
        )


class P4GeometryProvider(nn.Module):
    """Adapter that lets P4 GeometryHook call the learned P5 module."""

    def __init__(self, config: GeometryConfig, *, model_dim: int) -> None:
        super().__init__()
        self.config = config
        self.model_dim = model_dim
        effective_config = config.model_copy(update={"geometry_feature_dim": model_dim})
        self.geometry_module = LearnedGeometryModule(effective_config, input_dim=model_dim)
        self.output_projection = nn.Identity()

    def forward(
        self,
        model_input: AxiomModelInput,
        slot_states: Tensor,
        relation_graph: object | None = None,
    ) -> GeometryContext:
        del relation_graph
        graph_batch = graph_batch_from_model_input(model_input, self.config)
        node_states = _node_states_from_slots(slot_states, graph_batch)
        output = self.geometry_module(node_states=node_states, graph_batch=graph_batch)
        conditioning = _pool_nodes_to_records(
            output.conditioning_features,
            graph_batch,
            batch_size=model_input.batch_size,
        )
        observables = _observable_tensor(output.observables, batch_size=model_input.batch_size)
        return GeometryContext(
            observables=observables.to(device=slot_states.device, dtype=slot_states.dtype),
            conditioning=conditioning.to(device=slot_states.device, dtype=slot_states.dtype),
            diagnostics={
                **output.diagnostics.to_dict(),
                "enabled": self.config.mode != GeometryMode.OFF,
                "mode": self.config.mode.value,
                "p5_geometry_provider": True,
                "axc_out_fields": output.axc_out_fields,
            },
        )


def _ensure_loops(
    graph_batch: ClaimFieldGraphBatch,
    config: GeometryConfig,
) -> ClaimFieldGraphBatch:
    if graph_batch.loop_mask.numel() > 0:
        return graph_batch
    sample = sample_loops(graph_batch, config)
    return graph_batch.with_paths_and_loops(
        path_edge_index=sample.path_edge_index,
        path_mask=sample.path_mask,
        loop_path_index=sample.loop_path_index,
        loop_mask=sample.loop_mask,
    )


def _context_relation_ids(graph_batch: ClaimFieldGraphBatch) -> Tensor | None:
    no_relation_types = graph_batch.relation_type_ids.numel() == 0
    no_context_types = graph_batch.context_transition_type_ids.numel() == 0
    if no_relation_types or no_context_types:
        return None
    repeats = graph_batch.context_transition_type_ids.numel()
    return graph_batch.relation_type_ids[:1].expand(repeats)


def _context_lability(fiber_state: Tensor, graph_batch: ClaimFieldGraphBatch) -> Tensor:
    if graph_batch.context_edge_index.numel() == 0:
        return torch.zeros((), dtype=fiber_state.dtype, device=fiber_state.device)
    source = graph_batch.context_edge_index[0].clamp(max=max(0, fiber_state.shape[0] - 1))
    target = graph_batch.context_edge_index[1].clamp(max=max(0, fiber_state.shape[0] - 1))
    return (fiber_state.index_select(0, source) - fiber_state.index_select(0, target)).pow(2).mean()


def _node_degree_features(graph_batch: ClaimFieldGraphBatch, *, dtype: torch.dtype) -> Tensor:
    degree = torch.zeros((graph_batch.num_nodes, 4), dtype=dtype, device=graph_batch.device)
    for edge_index, column in (
        (graph_batch.relation_edge_index, 0),
        (graph_batch.context_edge_index, 2),
    ):
        if edge_index.numel() == 0:
            continue
        source = edge_index[0].clamp(max=max(0, graph_batch.num_nodes - 1))
        target = edge_index[1].clamp(max=max(0, graph_batch.num_nodes - 1))
        degree[:, column].scatter_add_(0, source, torch.ones_like(source, dtype=dtype))
        degree[:, column + 1].scatter_add_(0, target, torch.ones_like(target, dtype=dtype))
    return torch.log1p(degree)


def _empty_output(
    config: GeometryConfig,
    *,
    node_states: Tensor,
    graph_batch: ClaimFieldGraphBatch,
    mode: GeometryMode,
) -> GeometryOutput:
    conditioning = torch.zeros(
        (node_states.shape[0], config.geometry_feature_dim),
        dtype=node_states.dtype,
        device=node_states.device,
    )
    observables = empty_observables(device=node_states.device, dtype=node_states.dtype)
    return GeometryOutput(
        mode=mode,
        conditioning_features=conditioning,
        observables=observables,
        regularizer_terms=zero_regularizers(device=node_states.device, dtype=node_states.dtype),
        diagnostics=build_geometry_diagnostics(
            mode=mode,
            graph_batch=graph_batch,
            connection=None,
            observables=observables,
            no_loops=True,
            empty_graph=True,
        ),
        axc_out_fields=geometry_observables_to_axc_out_fields(
            mode=mode,
            enabled=mode != GeometryMode.OFF,
            observables=observables,
        ),
    )


def _node_states_from_slots(slot_states: Tensor, graph_batch: ClaimFieldGraphBatch) -> Tensor:
    pooled = slot_states.mean(dim=1)
    if graph_batch.num_nodes == 0:
        return torch.zeros(
            (0, slot_states.shape[-1]),
            dtype=slot_states.dtype,
            device=slot_states.device,
        )
    if graph_batch.node_to_claim_state is None:
        return pooled[:1].expand(graph_batch.num_nodes, -1)
    indices = graph_batch.node_to_claim_state.clamp(min=0, max=max(0, pooled.shape[0] - 1))
    return pooled.index_select(0, indices)


def _pool_nodes_to_records(
    node_features: Tensor,
    graph_batch: ClaimFieldGraphBatch,
    *,
    batch_size: int,
) -> Tensor:
    output = torch.zeros(
        (batch_size, node_features.shape[-1]),
        dtype=node_features.dtype,
        device=node_features.device,
    )
    counts = torch.zeros((batch_size, 1), dtype=node_features.dtype, device=node_features.device)
    if graph_batch.node_to_claim_state is None or graph_batch.node_to_claim_state.numel() == 0:
        return output
    indices = graph_batch.node_to_claim_state.clamp(min=0, max=max(0, batch_size - 1))
    output.index_add_(0, indices, node_features)
    counts.index_add_(
        0,
        indices,
        torch.ones(
            (indices.numel(), 1),
            dtype=node_features.dtype,
            device=node_features.device,
        ),
    )
    return output / counts.clamp_min(1.0)


def _observable_tensor(observables: GeometryObservables, *, batch_size: int) -> Tensor:
    values = torch.stack(
        [
            observables.curvature_score,
            observables.path_consistency_score,
            observables.context_lability_score,
            observables.loop_count.to(dtype=observables.curvature_score.dtype),
        ]
    )
    return values.reshape(1, -1).expand(batch_size, -1)
