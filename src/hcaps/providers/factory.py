"""Provider construction helpers."""

from __future__ import annotations

from hcaps.providers.base import ExtractionProvider
from hcaps.providers.config import ProviderEndpointConfig
from hcaps.providers.deterministic import DeterministicProvider
from hcaps.providers.openai_compatible import OpenAICompatibleProvider
from hcaps.providers.python_callable import PythonCallableProvider


def build_provider(config: ProviderEndpointConfig) -> ExtractionProvider:
    """Build a provider implementation from endpoint config."""

    if config.type == "deterministic":
        return DeterministicProvider(config)
    if config.type == "openai_compatible":
        return OpenAICompatibleProvider(config)
    return PythonCallableProvider(config)
