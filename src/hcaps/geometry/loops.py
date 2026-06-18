"""Bounded deterministic loop/path sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from hcaps.geometry.batching import ClaimFieldGraphBatch
from hcaps.geometry.config import GeometryConfig

TRIANGLE_LOOP_LENGTH = 3


@dataclass(frozen=True)
class LoopSample:
    path_edge_index: Tensor
    path_mask: Tensor
    loop_path_index: Tensor
    loop_mask: Tensor
    diagnostics: dict[str, int | bool]


def sample_loops(graph_batch: ClaimFieldGraphBatch, config: GeometryConfig) -> LoopSample:
    """Sample bounded provider/context loops deterministically."""

    device = graph_batch.device
    context_edges = [
        (
            int(graph_batch.context_edge_index[0, index].item()),
            int(graph_batch.context_edge_index[1, index].item()),
            index,
        )
        for index in range(graph_batch.context_edge_index.shape[1])
    ]
    edge_by_pair = {(source, target): edge_index for source, target, edge_index in context_edges}
    loop_edges: list[list[int]] = []
    for source, target, edge_index in context_edges:
        reverse = edge_by_pair.get((target, source))
        if reverse is not None:
            loop_edges.append([edge_index, reverse])
        if len(loop_edges) >= config.max_loops_per_batch:
            break

    if len(loop_edges) < config.max_loops_per_batch:
        loop_edges.extend(_triangle_loops(context_edges, edge_by_pair, config))

    loop_edges = _dedupe(loop_edges)[: config.max_loops_per_batch]
    path_edge_index = _single_edge_paths(graph_batch.context_edge_index.shape[1], device=device)
    path_mask = torch.ones_like(path_edge_index, dtype=torch.bool)
    if not loop_edges:
        width = max(1, config.max_loop_length)
        return LoopSample(
            path_edge_index=path_edge_index,
            path_mask=path_mask,
            loop_path_index=torch.zeros((0, width), dtype=torch.long, device=device),
            loop_mask=torch.zeros((0, width), dtype=torch.bool, device=device),
            diagnostics={
                "no_loops": True,
                "loop_count": 0,
                "bounded": True,
            },
        )

    width = max(1, config.max_loop_length)
    loop_path_index = torch.zeros((len(loop_edges), width), dtype=torch.long, device=device)
    loop_mask = torch.zeros((len(loop_edges), width), dtype=torch.bool, device=device)
    for row, edges in enumerate(loop_edges):
        for col, edge_index in enumerate(edges[:width]):
            loop_path_index[row, col] = edge_index
            loop_mask[row, col] = True
    return LoopSample(
        path_edge_index=path_edge_index,
        path_mask=path_mask,
        loop_path_index=loop_path_index,
        loop_mask=loop_mask,
        diagnostics={
            "no_loops": False,
            "loop_count": len(loop_edges),
            "bounded": True,
        },
    )


def _triangle_loops(
    context_edges: list[tuple[int, int, int]],
    edge_by_pair: dict[tuple[int, int], int],
    config: GeometryConfig,
) -> list[list[int]]:
    loops: list[list[int]] = []
    if config.max_loop_length < TRIANGLE_LOOP_LENGTH:
        return loops
    for source, mid, first in context_edges:
        for _, target, second in (edge for edge in context_edges if edge[0] == mid):
            third = edge_by_pair.get((target, source))
            if third is not None:
                loops.append([first, second, third])
            if len(loops) >= config.max_loops_per_batch:
                return loops
    return loops


def _single_edge_paths(edge_count: int, *, device: torch.device) -> Tensor:
    if edge_count == 0:
        return torch.zeros((0, 1), dtype=torch.long, device=device)
    return torch.arange(edge_count, dtype=torch.long, device=device).reshape(edge_count, 1)


def _dedupe(loop_edges: list[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    output: list[list[int]] = []
    for edges in loop_edges:
        key = tuple(edges)
        if key in seen:
            continue
        seen.add(key)
        output.append(edges)
    return output
