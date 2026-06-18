from __future__ import annotations

from pathlib import Path

import pytest

from hcaps.axt import AxtCompileConfig, compile_axt

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def p6_compiled_axt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("p6_axt") / "minimal.axt"
    compile_axt(
        AxtCompileConfig(
            input_path=ROOT / "examples" / "axf" / "v0_1" / "minimal_dataset.axp",
            output_path=output,
            field_registry_path=ROOT / "spec" / "FIELD_REGISTRY_V1.md",
            vocabulary_registry_path=ROOT / "spec" / "VOCABULARY_REGISTRY_V1.md",
            allow_all_without_split=True,
            max_text_length=32,
        ),
        force=True,
    )
    return output
