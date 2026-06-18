from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hcaps.axt import AxtBatch, AxtBatchCollator, AxtCompileConfig, AxtDataset, compile_axt
from hcaps.geometry import GeometryConfig
from hcaps.geometry.batching import ClaimFieldGraphBatch
from hcaps.geometry.graph import GeometryMasks, HyperedgeIncidence, edge_index_from_pairs

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def compiled_geometry_axt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("p5_geometry") / "minimal.axt"
    compile_axt(
        AxtCompileConfig(
            input_path=ROOT / "examples" / "axf" / "v0_1" / "minimal_dataset.axp",
            output_path=output,
            field_registry_path=ROOT / "spec" / "FIELD_REGISTRY_V1.md",
            vocabulary_registry_path=ROOT / "spec" / "VOCABULARY_REGISTRY_V1.md",
            allow_all_without_split=True,
            max_text_length=32,
        )
    )
    return output


@pytest.fixture
def geometry_axt_batch(compiled_geometry_axt: Path) -> AxtBatch:
    dataset = AxtDataset(compiled_geometry_axt)
    return AxtBatchCollator()([dataset[0], dataset[1]])


@pytest.fixture
def learned_geometry_config() -> GeometryConfig:
    return GeometryConfig.from_yaml(ROOT / "configs" / "geometry" / "geometry_learned_smoke.yaml")


@pytest.fixture
def tiny_loop_graph() -> ClaimFieldGraphBatch:
    device = torch.device("cpu")
    node_type_ids = torch.tensor([0, 2, 2], dtype=torch.long)
    relation_edge_index = edge_index_from_pairs([(0, 1), (0, 2)], device=device)
    context_edge_index = edge_index_from_pairs([(1, 2), (2, 1)], device=device)
    path_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    path_mask = torch.ones_like(path_edge_index, dtype=torch.bool)
    loop_path_index = torch.tensor([[0, 1]], dtype=torch.long)
    loop_mask = torch.ones_like(loop_path_index, dtype=torch.bool)
    masks = GeometryMasks(
        node_mask=torch.ones((3,), dtype=torch.bool),
        relation_edge_mask=torch.ones((2,), dtype=torch.bool),
        context_edge_mask=torch.ones((2,), dtype=torch.bool),
        path_mask=path_mask,
        loop_mask=loop_mask,
    )
    return ClaimFieldGraphBatch(
        node_ids=("claim:0", "context:a", "context:b"),
        node_type_ids=node_type_ids,
        node_to_claim_state=torch.tensor([0, 0, 0], dtype=torch.long),
        relation_edge_index=relation_edge_index,
        relation_type_ids=torch.tensor([1, 2], dtype=torch.long),
        relation_confidence=torch.ones((2,), dtype=torch.float32),
        context_node_ids=("context:a", "context:b"),
        context_edge_index=context_edge_index,
        context_transition_type_ids=torch.tensor([0, 1], dtype=torch.long),
        path_edge_index=path_edge_index,
        path_mask=path_mask,
        loop_path_index=loop_path_index,
        loop_mask=loop_mask,
        hyperedge_incidence=HyperedgeIncidence(
            hyperedge_id=torch.tensor([7, 7, 7], dtype=torch.long),
            node_id=torch.tensor([0, 1, 2], dtype=torch.long),
            role_id=torch.tensor([0, 1, 2], dtype=torch.long),
            hyperedge_type_id=torch.tensor([3, 3, 3], dtype=torch.long),
        ),
        temporal_features=torch.zeros((1, 8), dtype=torch.float32),
        lateral_context_features=torch.zeros((1, 8), dtype=torch.float32),
        provider_features=torch.zeros((1, 8), dtype=torch.float32),
        masks=masks,
    )
