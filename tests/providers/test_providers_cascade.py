from __future__ import annotations

import pytest

from hcaps.providers.base import ProviderRequest, ProviderResponse
from hcaps.providers.cache import ProviderCache
from hcaps.providers.cascade import ProviderCascade
from hcaps.providers.config import (
    ProviderCascadeConfig,
    ProviderEndpointConfig,
    ProviderGateConfig,
)
from hcaps.providers.deterministic import DeterministicProvider
from hcaps.providers.openai_compatible import OpenAICompatibleProvider


def test_deterministic_provider_returns_claim_candidates() -> None:
    config = ProviderEndpointConfig(provider_id="det", type="deterministic")
    provider = DeterministicProvider(config)
    response = provider.run(
        ProviderRequest(
            task="claim_extraction",
            input_text="GammaSuite uses containerized tests to measure repair outcomes.",
        )
    )

    assert response.provider_trace.provider_id == "det"
    assert response.normalized_output["claims"]
    assert response.normalized_output["claims"][0]["claim_type"] == "method_claim"


def test_openai_compatible_dry_run_never_calls_network() -> None:
    config = ProviderEndpointConfig(
        provider_id="local_dry_run",
        type="openai_compatible",
        provider_family="openai_compatible",
        provider_mode="local",
        base_url="http://127.0.0.1:9/v1",
        model="dry-run-model",
        dry_run=True,
    )
    response = OpenAICompatibleProvider(config).run(
        ProviderRequest(task="claim_extraction", input_text="AlphaBench improves compile success.")
    )

    assert response.confidence == 0.0
    assert response.raw_response["dry_run"] is True
    assert response.warnings


def test_provider_cascade_escalates_and_preserves_disagreement(tmp_path) -> None:
    primary_config = ProviderEndpointConfig(
        provider_id="primary_dry_run",
        type="openai_compatible",
        provider_family="openai_compatible",
        provider_mode="local",
        base_url="http://127.0.0.1:9/v1",
        model="dry-run-model",
        dry_run=True,
    )
    escalation_config = ProviderEndpointConfig(
        provider_id="deterministic_escalation",
        type="deterministic",
    )
    cascade = ProviderCascade(
        primary=OpenAICompatibleProvider(primary_config),
        primary_config=primary_config,
        escalation=DeterministicProvider(escalation_config),
        escalation_config=escalation_config,
        cache=ProviderCache(tmp_path / "cache"),
        cascade_config=ProviderCascadeConfig(
            enabled=True,
            gate=ProviderGateConfig(min_confidence=0.72, max_escalation_fraction=1.0),
        ),
    )

    result = cascade.run(
        ProviderRequest(
            task="claim_extraction",
            input_text="GammaSuite uses containerized tests to measure repair outcomes.",
        )
    )

    assert result.gate_decision.escalate is True
    assert result.merge.disagreements
    assert result.response.provider_trace.provider_id == "deterministic_escalation"
    assert result.response.provider_trace.merge_decision == "selected_escalation"


def test_provider_response_rejects_forbidden_active_label_keys() -> None:
    config = ProviderEndpointConfig(provider_id="det", type="deterministic")
    base_response = DeterministicProvider(config).run(
        ProviderRequest(task="claim_extraction", input_text="AlphaBench improves compile success.")
    )

    with pytest.raises(ValueError, match="forbidden active keys"):
        ProviderResponse(
            provider_trace=base_response.provider_trace,
            normalized_output={"is_true": True},
            confidence=0.1,
        )
