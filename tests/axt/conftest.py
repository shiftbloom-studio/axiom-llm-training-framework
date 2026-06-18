from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from hcaps.axt import AxtCompileConfig, compile_axt
from hcaps.corpus.builder import build_corpus
from hcaps.corpus.manifest import CorpusBuildConfig

ROOT = Path(__file__).resolve().parents[2]
MINIMAL_AXC = ROOT / "examples" / "axf" / "v0_1" / "minimal_capsules.axc"
MINIMAL_AXP = ROOT / "examples" / "axf" / "v0_1" / "minimal_dataset.axp"
FIELD_REGISTRY = ROOT / "spec" / "FIELD_REGISTRY_V1.md"
VOCAB_REGISTRY = ROOT / "spec" / "VOCABULARY_REGISTRY_V1.md"


@pytest.fixture
def compiled_minimal_axc(tmp_path: Path) -> Path:
    output = tmp_path / "minimal_axc.axt"
    compile_axt(
        AxtCompileConfig(
            input_path=MINIMAL_AXC,
            output_path=output,
            field_registry_path=FIELD_REGISTRY,
            vocabulary_registry_path=VOCAB_REGISTRY,
            allow_all_without_split=True,
            max_text_length=64,
        )
    )
    return output


@pytest.fixture
def compiled_minimal_axp(tmp_path: Path) -> Path:
    output = tmp_path / "minimal_axp.axt"
    compile_axt(
        AxtCompileConfig(
            input_path=MINIMAL_AXP,
            output_path=output,
            field_registry_path=FIELD_REGISTRY,
            vocabulary_registry_path=VOCAB_REGISTRY,
            allow_all_without_split=True,
            max_text_length=64,
        )
    )
    return output


@pytest.fixture(scope="session")
def p1_corpus_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("p1_corpus") / "ml_software"
    result = build_corpus(
        CorpusBuildConfig(
            corpus_id="p3_test_ml_software",
            input_path=ROOT / "examples" / "corpus" / "ml_software_benchmarks" / "sources",
            output_dir=output_dir,
            dataset_name="ml_software_benchmarks",
            domain=["benchmarking", "machine_learning", "software_engineering"],
            cutoff_date=date(2026, 1, 1),
        )
    )
    return result.package_path


@pytest.fixture
def compiled_p1_corpus(tmp_path: Path, p1_corpus_package: Path) -> Path:
    output = tmp_path / "p1_corpus.axt"
    compile_axt(
        AxtCompileConfig(
            input_path=p1_corpus_package,
            output_path=output,
            field_registry_path=FIELD_REGISTRY,
            vocabulary_registry_path=VOCAB_REGISTRY,
            allow_all_without_split=True,
            include_evaluation_references=True,
            max_text_length=96,
        )
    )
    return output
