from __future__ import annotations

from pathlib import Path

from hcaps.providers.base import ProviderRequest
from hcaps.providers.cache import ProviderCache
from hcaps.providers.config import ProviderEndpointConfig, load_provider_ingress_config
from hcaps.providers.deterministic import DeterministicProvider
from hcaps.providers.openai_compatible import OpenAICompatibleProvider


def test_provider_config_resolves_relative_cache_root(tmp_path: Path) -> None:
    config_path = tmp_path / "providers.yaml"
    config_path.write_text(
        """
providers:
  cache:
    root_path: cache/providers
    mode: cache_only
  primary:
    provider_id: local
    type: deterministic
""",
        encoding="utf-8",
    )

    config = load_provider_ingress_config(config_path)

    assert config.cache.root_path == tmp_path / "cache/providers"
    assert config.primary.provider_id == "local"
    assert config.primary.provider_mode == "deterministic"


def test_provider_cache_key_stable_and_replayable(tmp_path: Path) -> None:
    config = ProviderEndpointConfig(provider_id="det", type="deterministic")
    request = ProviderRequest(
        task="claim_extraction",
        input_text="GammaSuite uses containerized tests to measure repair outcomes.",
    )
    cache = ProviderCache(tmp_path / "cache")
    provider = DeterministicProvider(config)

    first = cache.replay_or_run(provider, config, request, "live")
    second = cache.replay_or_run(provider, config, request, "cache_only")

    assert first.normalized_output == second.normalized_output
    assert cache.request_hash(config, request) == cache.request_hash(config, request)
    assert cache.manifest().record_count == 1


def test_openai_dry_run_cache_does_not_store_secret_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AXIOM_TEST_PROVIDER_KEY", "secret-value-that-must-not-appear")
    config = ProviderEndpointConfig(
        provider_id="openai_compatible_dry_run",
        type="openai_compatible",
        provider_family="openai_compatible",
        provider_mode="local",
        base_url="http://localhost:8000/v1",
        model="local-extractor",
        api_key_env="AXIOM_TEST_PROVIDER_KEY",
        dry_run=True,
    )
    request = ProviderRequest(
        task="structured_view_generation",
        input_text="AlphaBench uses fixed temporal splits.",
    )
    cache = ProviderCache(tmp_path / "cache")
    response = cache.replay_or_run(OpenAICompatibleProvider(config), config, request, "refresh")
    cache_path = cache.path_for(config, request)

    assert response.raw_response["dry_run"] is True
    assert cache_path.exists()
    assert "secret-value-that-must-not-appear" not in cache_path.read_text(encoding="utf-8")
    assert "AXIOM_TEST_PROVIDER_KEY" in cache_path.read_text(encoding="utf-8")
