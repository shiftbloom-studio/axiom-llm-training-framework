"""Torch-ready geometry batch objects and AXT/P4 adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from hcaps.geometry.config import GeometryConfig
from hcaps.geometry.context import context_transition_type_id, deterministic_context_edges
from hcaps.geometry.graph import (
    NODE_TYPE_IDS,
    GeometryMasks,
    HyperedgeIncidence,
    edge_index_from_pairs,
)
from hcaps.model.config import AblationConfig, AxiomModelConfig
from hcaps.model.input_adapter import StructuredInputAdapter
from hcaps.model.types import AxiomModelInput


@dataclass(frozen=True)
class ClaimFieldGraphBatch:
    node_ids: tuple[str, ...]
    node_type_ids: Tensor
    node_to_claim_state: Tensor | None
    relation_edge_index: Tensor
    relation_type_ids: Tensor
    relation_confidence: Tensor | None
    context_node_ids: tuple[str, ...]
    context_edge_index: Tensor
    context_transition_type_ids: Tensor
    path_edge_index: Tensor
    path_mask: Tensor
    loop_path_index: Tensor
    loop_mask: Tensor
    hyperedge_incidence: HyperedgeIncidence | None
    temporal_features: Tensor | None
    lateral_context_features: Tensor | None
    provider_features: Tensor | None
    masks: GeometryMasks

    @property
    def device(self) -> torch.device:
        return self.node_type_ids.device

    @property
    def num_nodes(self) -> int:
        return int(self.node_type_ids.numel())

    @property
    def num_edges(self) -> int:
        return int(self.relation_edge_index.shape[1])

    @property
    def num_context_edges(self) -> int:
        return int(self.context_edge_index.shape[1])

    @classmethod
    def empty(cls, *, device: torch.device | None = None) -> ClaimFieldGraphBatch:
        real_device = device or torch.device("cpu")
        masks = GeometryMasks(
            node_mask=torch.zeros((0,), dtype=torch.bool, device=real_device),
            relation_edge_mask=torch.zeros((0,), dtype=torch.bool, device=real_device),
            context_edge_mask=torch.zeros((0,), dtype=torch.bool, device=real_device),
            path_mask=torch.zeros((0, 0), dtype=torch.bool, device=real_device),
            loop_mask=torch.zeros((0, 0), dtype=torch.bool, device=real_device),
        )
        return cls(
            node_ids=(),
            node_type_ids=torch.zeros((0,), dtype=torch.long, device=real_device),
            node_to_claim_state=None,
            relation_edge_index=torch.zeros((2, 0), dtype=torch.long, device=real_device),
            relation_type_ids=torch.zeros((0,), dtype=torch.long, device=real_device),
            relation_confidence=None,
            context_node_ids=(),
            context_edge_index=torch.zeros((2, 0), dtype=torch.long, device=real_device),
            context_transition_type_ids=torch.zeros((0,), dtype=torch.long, device=real_device),
            path_edge_index=torch.zeros((0, 0), dtype=torch.long, device=real_device),
            path_mask=torch.zeros((0, 0), dtype=torch.bool, device=real_device),
            loop_path_index=torch.zeros((0, 0), dtype=torch.long, device=real_device),
            loop_mask=torch.zeros((0, 0), dtype=torch.bool, device=real_device),
            hyperedge_incidence=None,
            temporal_features=None,
            lateral_context_features=None,
            provider_features=None,
            masks=masks,
        )

    def with_paths_and_loops(
        self,
        *,
        path_edge_index: Tensor,
        path_mask: Tensor,
        loop_path_index: Tensor,
        loop_mask: Tensor,
    ) -> ClaimFieldGraphBatch:
        return replace(
            self,
            path_edge_index=path_edge_index,
            path_mask=path_mask,
            loop_path_index=loop_path_index,
            loop_mask=loop_mask,
            masks=replace(self.masks, path_mask=path_mask, loop_mask=loop_mask),
        )


@dataclass(frozen=True)
class _NodeLayout:
    node_ids: list[str]
    node_type_ids: list[int]
    claim_node_indices: list[int]
    context_node_indices: list[int]
    provider_node_indices: list[int]
    source_node_indices: list[int]


@dataclass(frozen=True)
class _RelationLayout:
    pairs: list[tuple[int, int]]
    type_ids: list[int]
    confidence: list[float]


def graph_batch_from_axt_batch(
    batch: object,
    config: GeometryConfig,
    *,
    device: torch.device | str | None = None,
) -> ClaimFieldGraphBatch:
    adapter_config = AxiomModelConfig(
        model_dim=config.geometry_feature_dim,
        slot_dim=config.geometry_feature_dim,
        use_geometry_features=True,
        geometry_mode="geometry_provider_injected",
        ablations=AblationConfig(geometry_off=False),
    )
    model_input = StructuredInputAdapter(adapter_config, device=device)(batch)
    return graph_batch_from_model_input(model_input, config)


def graph_batch_from_model_input(
    model_input: AxiomModelInput,
    config: GeometryConfig,
) -> ClaimFieldGraphBatch:
    device = model_input.device
    batch_size = model_input.batch_size
    if batch_size == 0:
        return ClaimFieldGraphBatch.empty(device=device)

    nodes = _build_node_layout(model_input, batch_size)
    relations = _build_relation_layout(model_input, nodes, batch_size)

    context_pairs, transition_type_names = deterministic_context_edges(
        model_input,
        context_node_indices=nodes.context_node_indices,
        provider_node_indices=nodes.provider_node_indices,
        include_provider=config.use_provider_context,
        include_disagreement=config.use_provider_disagreement_edges,
    )
    context_transition_types = [context_transition_type_id(name) for name in transition_type_names]

    lateral_features = _group_feature_matrix(
        model_input.group("lateral_context"),
        batch_size,
        device,
    )
    provider_features = _group_feature_matrix(
        model_input.group("provider_context"),
        batch_size,
        device,
    )
    temporal_features = _group_feature_matrix(model_input.group("temporal"), batch_size, device)
    hyperedge_incidence = _hyperedge_incidence_from_model_input(
        model_input,
        claim_node_indices=nodes.claim_node_indices,
        context_node_indices=nodes.context_node_indices,
        source_node_indices=nodes.source_node_indices,
    )

    relation_edge_index = edge_index_from_pairs(relations.pairs, device=device)
    context_edge_index = edge_index_from_pairs(context_pairs, device=device)
    path_edge_index = (
        torch.arange(context_edge_index.shape[1], dtype=torch.long, device=device).reshape(-1, 1)
        if context_edge_index.shape[1] > 0
        else torch.zeros((0, 1), dtype=torch.long, device=device)
    )
    path_mask = torch.ones_like(path_edge_index, dtype=torch.bool)
    loop_path_index = torch.zeros(
        (0, max(1, config.max_loop_length)),
        dtype=torch.long,
        device=device,
    )
    loop_mask = torch.zeros_like(loop_path_index, dtype=torch.bool)

    masks = GeometryMasks(
        node_mask=torch.ones((len(nodes.node_ids),), dtype=torch.bool, device=device),
        relation_edge_mask=torch.ones((len(relations.pairs),), dtype=torch.bool, device=device),
        context_edge_mask=torch.ones((len(context_pairs),), dtype=torch.bool, device=device),
        path_mask=path_mask,
        loop_mask=loop_mask,
    )
    return ClaimFieldGraphBatch(
        node_ids=tuple(nodes.node_ids),
        node_type_ids=torch.tensor(nodes.node_type_ids, dtype=torch.long, device=device),
        node_to_claim_state=torch.tensor(
            [index // 4 for index in range(len(nodes.node_ids))],
            dtype=torch.long,
            device=device,
        ),
        relation_edge_index=relation_edge_index,
        relation_type_ids=torch.tensor(relations.type_ids, dtype=torch.long, device=device),
        relation_confidence=torch.tensor(relations.confidence, dtype=torch.float32, device=device),
        context_node_ids=tuple(nodes.node_ids[index] for index in nodes.context_node_indices),
        context_edge_index=context_edge_index,
        context_transition_type_ids=torch.tensor(
            context_transition_types,
            dtype=torch.long,
            device=device,
        ),
        path_edge_index=path_edge_index,
        path_mask=path_mask,
        loop_path_index=loop_path_index,
        loop_mask=loop_mask,
        hyperedge_incidence=hyperedge_incidence,
        temporal_features=temporal_features,
        lateral_context_features=lateral_features,
        provider_features=provider_features,
        masks=masks,
    )


def _build_node_layout(model_input: AxiomModelInput, batch_size: int) -> _NodeLayout:
    node_ids: list[str] = []
    node_type_ids: list[int] = []
    claim_node_indices: list[int] = []
    context_node_indices: list[int] = []
    provider_node_indices: list[int] = []
    source_node_indices: list[int] = []
    ids = model_input.group("ids")
    for index in range(batch_size):
        claim_node_indices.append(len(node_ids))
        node_ids.append(f"claim_state:{_scalar(ids, 'claim_state_index', index)}")
        node_type_ids.append(NODE_TYPE_IDS["claim_state"])

        context_node_indices.append(len(node_ids))
        node_ids.append(f"context:{_scalar(ids, 'context_index', index)}")
        node_type_ids.append(NODE_TYPE_IDS["context"])

        provider_node_indices.append(len(node_ids))
        node_ids.append(f"provider:{_scalar(ids, 'provider_trace_index', index)}")
        node_type_ids.append(NODE_TYPE_IDS["provider"])

        source_node_indices.append(len(node_ids))
        node_ids.append(f"source:{_scalar(ids, 'source_index', index)}")
        node_type_ids.append(NODE_TYPE_IDS["source"])
    return _NodeLayout(
        node_ids=node_ids,
        node_type_ids=node_type_ids,
        claim_node_indices=claim_node_indices,
        context_node_indices=context_node_indices,
        provider_node_indices=provider_node_indices,
        source_node_indices=source_node_indices,
    )


def _build_relation_layout(
    model_input: AxiomModelInput,
    nodes: _NodeLayout,
    batch_size: int,
) -> _RelationLayout:
    pairs: list[tuple[int, int]] = []
    type_ids: list[int] = []
    confidence: list[float] = []
    for index, claim_node in enumerate(nodes.claim_node_indices):
        pairs.extend(
            [
                (claim_node, nodes.context_node_indices[index]),
                (claim_node, nodes.provider_node_indices[index]),
                (claim_node, nodes.source_node_indices[index]),
            ]
        )
        type_ids.extend([0, 1, 2])
        confidence.extend([1.0, 1.0, 1.0])
        if batch_size <= 1:
            continue
        pairs.append((claim_node, nodes.claim_node_indices[(index + 1) % batch_size]))
        type_ids.append(_scalar(model_input.group("relations"), "relation_type_values", index))
        confidence.append(
            _float_scalar(model_input.group("relations"), "relation_confidence_values", index)
        )
    return _RelationLayout(pairs=pairs, type_ids=type_ids, confidence=confidence)


def _hyperedge_incidence_from_model_input(
    model_input: AxiomModelInput,
    *,
    claim_node_indices: list[int],
    context_node_indices: list[int],
    source_node_indices: list[int],
) -> HyperedgeIncidence | None:
    if not claim_node_indices:
        return None
    hyperedge_ids: list[int] = []
    node_ids: list[int] = []
    role_ids: list[int] = []
    type_ids: list[int] = []
    for index, claim_node in enumerate(claim_node_indices):
        hyperedge_id = _scalar(model_input.group("relations"), "hyperedge_id", index)
        if hyperedge_id < 0:
            hyperedge_id = index
        for role, node in enumerate(
            (claim_node, context_node_indices[index], source_node_indices[index])
        ):
            hyperedge_ids.append(hyperedge_id)
            node_ids.append(node)
            role_ids.append(role)
            type_ids.append(_scalar(model_input.group("relations"), "relation_type_values", index))
    device = model_input.device
    return HyperedgeIncidence(
        hyperedge_id=torch.tensor(hyperedge_ids, dtype=torch.long, device=device),
        node_id=torch.tensor(node_ids, dtype=torch.long, device=device),
        role_id=torch.tensor(role_ids, dtype=torch.long, device=device),
        hyperedge_type_id=torch.tensor(type_ids, dtype=torch.long, device=device),
    )


def _group_feature_matrix(
    group: dict[str, Tensor],
    batch_size: int,
    device: torch.device,
    *,
    width: int = 8,
) -> Tensor:
    features = torch.zeros((batch_size, width), dtype=torch.float32, device=device)
    for cursor, key in enumerate(sorted(group)):
        if cursor >= width:
            break
        tensor = group[key]
        values = _as_batch_vector(tensor, batch_size=batch_size, device=device)
        features[:, cursor] = torch.sign(values) * torch.log1p(torch.abs(values)) / 32.0
    return features


def _scalar(group: dict[str, Tensor], key: str, index: int) -> int:
    if key not in group:
        return -1
    values = _as_batch_vector(group[key], batch_size=index + 1, device=group[key].device)
    return int(values[index].item()) if index < values.numel() else -1


def _float_scalar(group: dict[str, Tensor], key: str, index: int) -> float:
    if key not in group:
        return 0.0
    values = _as_batch_vector(group[key], batch_size=index + 1, device=group[key].device)
    return float(values[index].item()) if index < values.numel() else 0.0


def _as_batch_vector(value: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
    tensor = value.to(device=device)
    if tensor.ndim == 0:
        return tensor.reshape(1).expand(batch_size).to(dtype=torch.float32)
    if tensor.ndim == 1:
        if tensor.shape[0] >= batch_size:
            return tensor[:batch_size].to(dtype=torch.float32)
        return tensor[:1].expand(batch_size).to(dtype=torch.float32)
    reshaped = tensor.reshape(tensor.shape[0], -1)
    if reshaped.shape[0] >= batch_size:
        return reshaped[:batch_size, 0].to(dtype=torch.float32)
    return reshaped[:1, 0].expand(batch_size).to(dtype=torch.float32)
