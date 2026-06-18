"""Provider ingress exceptions."""

from __future__ import annotations


class ProviderError(RuntimeError):
    """Base provider ingress error."""


class ProviderConfigError(ProviderError):
    """Raised when provider configuration is invalid."""


class ProviderCacheMissError(ProviderError):
    """Raised when cache-only replay cannot find a cached response."""


ProviderCacheMiss = ProviderCacheMissError


class ProviderExecutionError(ProviderError):
    """Raised when a provider cannot complete a request."""


class ProviderMergeError(ProviderError):
    """Raised when provider outputs cannot be merged deterministically."""
