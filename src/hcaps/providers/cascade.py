"""Config-driven local-first provider cascade."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hcaps.providers.base import ExtractionProvider, ProviderRequest, ProviderResponse
from hcaps.providers.cache import ProviderCache
from hcaps.providers.config import ProviderCascadeConfig, ProviderEndpointConfig
from hcaps.providers.merge import MergeResult, merge_provider_outputs


class EscalationDecision(BaseModel):
    """Deterministic decision emitted by the provider gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    escalate: bool
    reason_codes: list[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0)


class CascadeRunResult(BaseModel):
    """Provider cascade result with gate and disagreement traces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response: ProviderResponse
    gate_decision: EscalationDecision
    merge: MergeResult


class ProviderCascade:
    """Run primary extraction, optional escalation, and deterministic merge."""

    def __init__(
        self,
        *,
        primary: ExtractionProvider,
        primary_config: ProviderEndpointConfig,
        cache: ProviderCache,
        cascade_config: ProviderCascadeConfig,
        escalation: ExtractionProvider | None = None,
        escalation_config: ProviderEndpointConfig | None = None,
    ) -> None:
        self.primary = primary
        self.primary_config = primary_config
        self.escalation = escalation
        self.escalation_config = escalation_config
        self.cache = cache
        self.config = cascade_config

    def run(self, request: ProviderRequest) -> CascadeRunResult:
        primary_response = self.cache.replay_or_run(
            self.primary,
            self.primary_config,
            request,
            self.primary_config.cache_mode,
        )
        decision = self.decide(primary_response)
        escalation_response: ProviderResponse | None = None
        if (
            decision.escalate
            and self.config.enabled
            and self.escalation is not None
            and self.escalation_config is not None
        ):
            escalation_response = self.cache.replay_or_run(
                self.escalation,
                self.escalation_config,
                request,
                self.escalation_config.cache_mode,
            )
            escalation_response = escalation_response.model_copy(
                update={
                    "provider_trace": escalation_response.provider_trace.model_copy(
                        update={"escalation_reason": ",".join(decision.reason_codes)}
                    )
                }
            )
        merge = merge_provider_outputs(
            primary_response,
            escalation_response,
            merge_policy=self.config.merge_policy,
        )
        return CascadeRunResult(response=merge.response, gate_decision=decision, merge=merge)

    def decide(self, response: ProviderResponse) -> EscalationDecision:
        if not self.config.enabled or self.config.gate.max_escalation_fraction <= 0.0:
            return EscalationDecision(escalate=False, reason_codes=[], score=0.0)
        reasons: list[str] = []
        confidence = response.confidence
        if confidence is not None and confidence < self.config.gate.min_confidence:
            reasons.append("low_confidence")
        if self.config.gate.escalate_on_warnings and response.warnings:
            reasons.append("provider_warning")
        if _has_high_impact_claim(response, self.config.gate.high_impact_claim_types):
            reasons.append("high_impact_claim_type")
        score = min(1.0, len(reasons) / 3.0)
        return EscalationDecision(escalate=bool(reasons), reason_codes=reasons, score=score)


def _has_high_impact_claim(response: ProviderResponse, claim_types: list[str]) -> bool:
    if not claim_types:
        return False
    claims = response.normalized_output.get("claims", [])
    if not isinstance(claims, list):
        return False
    return any(
        isinstance(claim, dict) and claim.get("claim_type") in claim_types for claim in claims
    )
