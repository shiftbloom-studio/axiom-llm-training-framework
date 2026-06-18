"""OpenAI-compatible provider client for data construction only."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import orjson

from hcaps.extraction.contracts import ProviderTrace
from hcaps.providers.base import ProviderRequest, ProviderResponse, hash_normalized_output
from hcaps.providers.config import ProviderEndpointConfig
from hcaps.providers.errors import ProviderExecutionError


class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible chat-completions provider.

    This client is for corpus construction only. Tests should use dry-run or cache-only
    paths and must not require network access.
    """

    def __init__(self, config: ProviderEndpointConfig) -> None:
        self.config = config
        self.provider_id = config.provider_id

    def run(self, request: ProviderRequest) -> ProviderResponse:
        if self.config.dry_run:
            return self._dry_run_response(request)
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"  # type: ignore[union-attr]
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return schema-compatible JSON only. Do not emit truth labels.",
                },
                {"role": "user", "content": request.input_text},
            ],
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            api_key = os.getenv(self.config.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        raw_response = self._post_json(url, payload, headers)
        normalized = _normalize_openai_response(raw_response)
        output_hash = hash_normalized_output(normalized)
        return ProviderResponse(
            provider_trace=self._trace(request, output_hash, normalized.get("confidence")),
            normalized_output=normalized,
            confidence=_optional_float(normalized.get("confidence")),
            warnings=[],
            raw_response=raw_response,
        )

    def _dry_run_response(self, request: ProviderRequest) -> ProviderResponse:
        normalized = {
            "claims": [],
            "relations": [],
            "views": {},
            "dry_run": True,
        }
        output_hash = hash_normalized_output(normalized)
        return ProviderResponse(
            provider_trace=self._trace(request, output_hash, 0.0),
            normalized_output=normalized,
            confidence=0.0,
            warnings=["openai-compatible provider dry-run; no network request made"],
            raw_response={"dry_run": True, "model": self.config.model},
        )

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        body = orjson.dumps(payload)
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                request = Request(url, data=body, headers=headers, method="POST")
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    data = response.read()
                result = orjson.loads(data)
                if not isinstance(result, dict):
                    raise ProviderExecutionError("provider response must be a JSON object")
                return result
            except (HTTPError, URLError, TimeoutError, OSError, ProviderExecutionError) as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(min(2.0, 0.25 * (attempt + 1)))
        raise ProviderExecutionError(f"provider request failed: {last_error}") from last_error

    def _trace(
        self,
        request: ProviderRequest,
        output_hash: str,
        confidence: Any,
    ) -> ProviderTrace:
        return ProviderTrace(
            provider_id=self.config.provider_id,
            provider_type=self.config.type,
            provider_family=self.config.provider_family,
            provider_mode=self.config.provider_mode,
            provider_model=self.config.model,
            extraction_task=request.task,
            prompt_template_version=request.template_version,
            input_hash=request.input_hash,
            output_hash=output_hash,
            confidence=_optional_float(confidence),
        )


def _normalize_openai_response(raw_response: dict[str, Any]) -> dict[str, Any]:
    content = raw_response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            return {"raw_text": content, "confidence": 0.0}
        if isinstance(decoded, dict):
            return decoded
    return {"raw_text": str(content), "confidence": 0.0}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except TypeError, ValueError:
        return None
