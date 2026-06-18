"""Learned claim-field geometry for Axiom P5."""

from hcaps.geometry.batching import ClaimFieldGraphBatch, graph_batch_from_axt_batch
from hcaps.geometry.config import GeometryConfig, GeometryMode
from hcaps.geometry.connection import LearnedConnection
from hcaps.geometry.graph import HyperedgeIncidence, expand_hyperedges_to_pairwise
from hcaps.geometry.module import (
    GeometryOutput,
    LearnedGeometryModule,
    NoGeometryModule,
    NonGeometricContextMixer,
    P4GeometryProvider,
    P4NonGeometricContextProvider,
)

__all__ = [
    "ClaimFieldGraphBatch",
    "GeometryConfig",
    "GeometryMode",
    "GeometryOutput",
    "HyperedgeIncidence",
    "LearnedConnection",
    "LearnedGeometryModule",
    "NoGeometryModule",
    "NonGeometricContextMixer",
    "P4GeometryProvider",
    "P4NonGeometricContextProvider",
    "expand_hyperedges_to_pairwise",
    "graph_batch_from_axt_batch",
]
