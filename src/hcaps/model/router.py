"""Epistemic router for computational routing signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from hcaps.model.config import AxiomModelConfig
from hcaps.model.geometry_hooks import GeometryContext
from hcaps.model.types import AxiomModelInput, CoreOutput


@dataclass(frozen=True)
class RouterOutput:
    slot_states: Tensor
    latent_state: Tensor
    head_weights: Tensor
    loss_term_weights: dict[str, Tensor]
    diagnostics: dict[str, Any]


class EpistemicRouter(nn.Module):
    """Route epistemic work without emitting truth/correctness labels."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.config = config
        self.routing = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 6),
        )
        self.scalar_gate = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        core_output: CoreOutput,
        model_input: AxiomModelInput,
        geometry_context: GeometryContext,
    ) -> RouterOutput:
        del model_input, geometry_context
        mode = self.config.effective_router_mode()
        logits = self.routing(
            torch.cat([core_output.router_state, core_output.context_state], dim=-1)
        )
        head_weights = torch.softmax(logits, dim=-1)
        loss_weights = {
            "relation_head_weight": head_weights[:, 0],
            "provenance_head_weight": head_weights[:, 1],
            "future_head_weight": head_weights[:, 2],
            "geometry_head_weight": head_weights[:, 3],
            "uncertainty_gate": head_weights[:, 4],
            "revision_gate": head_weights[:, 5],
        }

        if mode == "router_off":
            return RouterOutput(
                slot_states=core_output.slot_states,
                latent_state=core_output.router_state,
                head_weights=torch.ones_like(head_weights) / head_weights.shape[-1],
                loss_term_weights=loss_weights,
                diagnostics={
                    "enabled": False,
                    "mode": mode,
                    "forbidden_truth_outputs": False,
                },
            )

        gate = self.scalar_gate(core_output.router_state)
        if mode in {"scalar_gate", "head_gate", "expert_gate"}:
            slot_states = core_output.slot_states * (1.0 + gate.unsqueeze(1))
            latent_state = core_output.router_state * (1.0 + gate)
        else:
            slot_states = core_output.slot_states
            latent_state = core_output.router_state
        return RouterOutput(
            slot_states=slot_states,
            latent_state=latent_state,
            head_weights=head_weights,
            loss_term_weights=loss_weights,
            diagnostics={
                "enabled": True,
                "mode": mode,
                "routing_logits_shape": tuple(logits.shape),
                "head_weights_shape": tuple(head_weights.shape),
                "forbidden_truth_outputs": False,
                "finite": bool(torch.isfinite(head_weights).all().item()),
            },
        )
