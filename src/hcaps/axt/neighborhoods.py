"""Relation-neighborhood tensor compilation."""

from __future__ import annotations

from typing import Any

from .schema import MISSING_INT, TensorGroup, as_f32, as_i64, stable_int64
from .vocab import VocabularyRegistry


def compile_relation_neighborhoods(
    capsules: list[Any],
    family_to_index: dict[str, int],
    vocab: VocabularyRegistry,
    *,
    include_relation_neighborhoods: bool,
) -> TensorGroup:
    """Compile pairwise relation neighborhoods while reserving hypergraph fields."""

    node_values: list[int] = []
    edge_type_values: list[int] = []
    edge_features: list[float] = []
    node_offsets = [0]
    edge_offsets = [0]
    masks: list[int] = []

    for capsule in capsules:
        relations = [] if not include_relation_neighborhoods else capsule.relations
        for relation in relations:
            node_values.append(family_to_index.get(relation.target_claim_family_id, MISSING_INT))
            edge_type_values.append(vocab.lookup("relation_type", relation.relation_type))
            edge_features.append(
                float(relation.confidence) if relation.confidence is not None else 0.0
            )
        node_offsets.append(len(node_values))
        edge_offsets.append(len(edge_type_values))
        masks.append(1 if relations else 0)

    return {
        "neighbor_node_values": as_i64(node_values),
        "neighbor_edge_type_values": as_i64(edge_type_values),
        "edge_feature_values": as_f32(edge_features),
        "node_offsets": as_i64(node_offsets),
        "edge_offsets": as_i64(edge_offsets),
        "neighborhood_masks": as_i64(masks),
        "hyperedge_id": as_i64([]),
        "hyperedge_type": as_i64([]),
        "hyperedge_incidence_values": as_i64([]),
        "hyperedge_offsets": as_i64([0] * (len(capsules) + 1)),
        "hypergraph_reserved": as_i64([1]),
        "neighborhood_schema_ref": as_i64([stable_int64("pairwise_with_reserved_hypergraph_path")]),
    }
