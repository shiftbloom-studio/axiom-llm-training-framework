from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
from typer.testing import CliRunner

from hcaps.cli import app
from hcaps.corpus.builder import build_corpus
from hcaps.corpus.gold import export_gold_candidates
from hcaps.corpus.manifest import CorpusBuildConfig
from hcaps.format.package import validate_package
from hcaps.providers.config import (
    ProviderCacheConfig,
    ProviderEndpointConfig,
    ProviderIngressConfig,
)

FIXTURE_SOURCES = (
    Path(__file__).parents[2] / "examples" / "corpus" / "ml_software_benchmarks" / "sources"
)
FORBIDDEN_KEYS = {
    "truth",
    "is_true",
    "correct",
    "is_correct",
    "factuality",
    "ground_truth",
    "label_truth",
    "proven_true",
}


def test_build_corpus_emits_p1_artifacts_and_valid_axp(tmp_path: Path) -> None:
    config = _corpus_config(tmp_path)

    result = build_corpus(config)

    expected = {
        "corpus_manifest",
        "provider_cache_manifest",
        "extraction_trace",
        "provider_disagreements",
        "source_registry",
        "claim_families",
        "relation_candidates",
        "negative_pools",
        "gold_candidates",
        "leakage_report",
        "license_report",
        "quality_report",
        "package",
    }
    assert expected <= set(result.artifact_paths)
    assert result.manifest.claim_family_count > 0
    assert validate_package(result.package_path).ok
    assert (result.package_path / "data" / "contexts.axctx").exists()
    assert (result.package_path / "manifests" / "provider_report.json").exists()

    family_records = _read_jsonl(result.artifact_paths["claim_families"])
    assert family_records
    assert family_records[0]["provider_traces"]["claim_extraction"]["provider_id"] == "det"
    assert family_records[0]["structured_views"]
    assert "epistemic_proxies" in family_records[0]
    assert not _forbidden_keys_in(family_records)

    source_registry = _read_jsonl(result.artifact_paths["source_registry"])
    assert source_registry[0]["license"] == "Apache-2.0"
    negative_pools = _read_jsonl(result.artifact_paths["negative_pools"])
    assert {record["pool_type"] for record in negative_pools}
    gold_candidates = _read_jsonl(result.artifact_paths["gold_candidates"])
    assert gold_candidates[0]["gold_reference"]


def test_corpus_cli_build_and_gold_export(tmp_path: Path) -> None:
    config_path = tmp_path / "corpus.yaml"
    output_dir = tmp_path / "corpus_out"
    cache_dir = tmp_path / "cache"
    config_path.write_text(
        f"""
corpus_id: axiom.fixture.cli
corpus_version: "0.1.0"
domain: [machine_learning, software_engineering, benchmarking]
dataset_name: cli_fixture
input_path: {FIXTURE_SOURCES}
output_dir: {output_dir}
cutoff_date: 2026-01-01
max_chunk_chars: 420
providers:
  cache:
    root_path: {cache_dir}
    mode: live
  primary:
    provider_id: det
    type: deterministic
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["corpus", "build", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert (output_dir / "corpus_manifest.json").exists()

    inspect_result = runner.invoke(app, ["corpus", "inspect", str(output_dir / "dataset.axp")])
    assert inspect_result.exit_code == 0, inspect_result.output

    cache_result = runner.invoke(app, ["corpus", "cache", "inspect", str(cache_dir)])
    assert cache_result.exit_code == 0, cache_result.output

    gold_output = tmp_path / "gold.jsonl"
    gold_result = runner.invoke(
        app,
        [
            "corpus",
            "gold",
            "export",
            "--input",
            str(output_dir / "dataset.axp"),
            "--output",
            str(gold_output),
        ],
    )
    assert gold_result.exit_code == 0, gold_result.output
    assert gold_output.exists()
    exported = export_gold_candidates(output_dir / "dataset.axp", tmp_path / "gold_again.jsonl")
    assert exported.exists()


def _corpus_config(tmp_path: Path) -> CorpusBuildConfig:
    return CorpusBuildConfig(
        corpus_id="axiom.ml_software_benchmark_claims.test",
        corpus_version="0.1.0",
        domain=["machine_learning", "software_engineering", "benchmarking"],
        input_path=FIXTURE_SOURCES,
        output_dir=tmp_path / "corpus",
        cutoff_date=None,
        max_chunk_chars=420,
        providers=ProviderIngressConfig(
            cache=ProviderCacheConfig(root_path=tmp_path / "cache", mode="live"),
            primary=ProviderEndpointConfig(provider_id="det", type="deterministic"),
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _forbidden_keys_in(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_KEYS:
                found.add(key)
            found.update(_forbidden_keys_in(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_forbidden_keys_in(nested))
    return found
