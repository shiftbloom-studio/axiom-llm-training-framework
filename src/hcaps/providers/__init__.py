"""Provider ingress for Axiom corpus construction.

`hcaps` is a legacy internal package name. Public project identity is Axiom.
Providers are allowed only for data construction/substrate harvesting, not for
model training, evaluation, scoring, or benchmark judging.
"""

from hcaps.providers.base import ExtractionProvider, ProviderRequest, ProviderResponse
from hcaps.providers.cache import ProviderCache, ProviderCacheRecord
from hcaps.providers.cascade import ProviderCascade
from hcaps.providers.config import (
    ProviderEndpointConfig,
    ProviderIngressConfig,
    load_provider_ingress_config,
)
from hcaps.providers.deterministic import DeterministicProvider
from hcaps.providers.factory import build_provider
from hcaps.providers.openai_compatible import OpenAICompatibleProvider
from hcaps.providers.python_callable import PythonCallableProvider

__all__ = [
    "DeterministicProvider",
    "ExtractionProvider",
    "OpenAICompatibleProvider",
    "ProviderCache",
    "ProviderCacheRecord",
    "ProviderCascade",
    "ProviderEndpointConfig",
    "ProviderIngressConfig",
    "ProviderRequest",
    "ProviderResponse",
    "PythonCallableProvider",
    "build_provider",
    "load_provider_ingress_config",
]
