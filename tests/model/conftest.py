from __future__ import annotations

from pathlib import Path

import pytest

from hcaps.axt import AxtBatch, AxtBatchCollator, AxtCompileConfig, AxtDataset, compile_axt
from hcaps.model import AxiomModelConfig

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def compiled_model_axt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("p4_model") / "minimal.axt"
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
def smoke_config() -> AxiomModelConfig:
    return AxiomModelConfig.from_yaml(ROOT / "configs" / "model" / "structured_native_smoke.yaml")


@pytest.fixture
def axt_batch(compiled_model_axt: Path) -> AxtBatch:
    dataset = AxtDataset(compiled_model_axt)
    return AxtBatchCollator()([dataset[0], dataset[1]])
