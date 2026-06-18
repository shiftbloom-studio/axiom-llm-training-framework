"""Python-callable extraction provider."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

from hcaps.extraction.contracts import ProviderTrace
from hcaps.providers.base import ProviderRequest, ProviderResponse, hash_normalized_output
from hcaps.providers.config import ProviderEndpointConfig
from hcaps.providers.errors import ProviderExecutionError


class PythonCallableProvider:
    """Provider backed by a local import path such as module:function."""

    def __init__(self, config: ProviderEndpointConfig) -> None:
        self.config = config
        self.provider_id = config.provider_id
        self._callable = _load_callable(config.callable_path)

    def run(self, request: ProviderRequest) -> ProviderResponse:
        try:
            result = self._callable(request)
        except Exception as exc:
            raise ProviderExecutionError(f"python callable provider failed: {exc}") from exc
        if isinstance(result, ProviderResponse):
            return result
        if not isinstance(result, dict):
            msg = "python callable provider must return ProviderResponse or dict"
            raise ProviderExecutionError(msg)
        normalized = dict(result.get("normalized_output", result))
        confidence = _optional_float(result.get("confidence"))
        warnings = list(result.get("warnings", []))
        raw_response = dict(result.get("raw_response", {"callable_result": normalized}))
        output_hash = hash_normalized_output(normalized)
        return ProviderResponse(
            provider_trace=ProviderTrace(
                provider_id=self.config.provider_id,
                provider_type=self.config.type,
                provider_family=self.config.provider_family,
                provider_mode=self.config.provider_mode,
                provider_model=self.config.model,
                extraction_task=request.task,
                prompt_template_version=request.template_version,
                input_hash=request.input_hash,
                output_hash=output_hash,
                confidence=confidence,
            ),
            normalized_output=normalized,
            confidence=confidence,
            warnings=warnings,
            raw_response=raw_response,
        )


def _load_callable(path: str | None) -> Callable[[ProviderRequest], Any]:
    if not path or ":" not in path:
        msg = "callable_path must use module:function syntax"
        raise ProviderExecutionError(msg)
    module_name, function_name = path.split(":", 1)
    module = import_module(module_name)
    candidate = getattr(module, function_name, None)
    if not callable(candidate):
        msg = f"callable_path does not resolve to a callable: {path}"
        raise ProviderExecutionError(msg)
    return cast("Callable[[ProviderRequest], Any]", candidate)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except TypeError, ValueError:
        return None
