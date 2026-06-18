"""Geometry hook interfaces for P4 and P5 handoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from torch import Tensor, nn

from hcaps.model.config import AxiomModelConfig
from hcaps.model.features import GroupFeatureProjector
from hcaps.model.types import AxiomModelInput


@dataclass(frozen=True)
class GeometryContext:
    observables: Tensor | None
    conditioning: Tensor | None
    diagnostics: dict[str, Any]
    regularizer_terms: dict[str, Tensor] | None = None


class GeometryProviderProtocol(Protocol):
    def forward(
        self,
        model_input: AxiomModelInput,
        slot_states: Tensor,
        relation_graph: object | None = None,
    ) -> GeometryContext:
        """Return geometry conditioning without leaking raw gauge matrices."""


class NullGeometryProvider(nn.Module):
    """Neutral geometry provider used for geometry_off ablations."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.model_dim = model_dim

    def forward(
        self,
        model_input: AxiomModelInput,
        slot_states: Tensor,
        relation_graph: object | None = None,
    ) -> GeometryContext:
        del relation_graph
        conditioning = torch.zeros(
            (model_input.batch_size, self.model_dim),
            dtype=slot_states.dtype,
            device=slot_states.device,
        )
        return GeometryContext(
            observables=None,
            conditioning=conditioning,
            diagnostics={
                "enabled": False,
                "mode": "geometry_off",
                "gauge_invariant_only": True,
                "raw_gauge_matrices_emitted": False,
            },
        )


class AxtGeometryFeatureProvider(nn.Module):
    """Project P3 gauge-invariant geometry observable slots into conditioning."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.projector = GroupFeatureProjector(config)

    def forward(
        self,
        model_input: AxiomModelInput,
        slot_states: Tensor,
        relation_graph: object | None = None,
    ) -> GeometryContext:
        del relation_graph
        conditioning = self.projector(
            model_input.group("geometry_observables"),
            batch_size=model_input.batch_size,
            device=model_input.device,
        )
        observables = conditioning[:, : min(conditioning.shape[-1], 8)]
        return GeometryContext(
            observables=observables,
            conditioning=conditioning.to(dtype=slot_states.dtype),
            diagnostics={
                "enabled": True,
                "mode": "geometry_features_from_axt",
                "gauge_invariant_only": True,
                "raw_gauge_matrices_emitted": False,
                "observable_shape": tuple(observables.shape),
            },
        )


class GeometryHook(nn.Module):
    """Switch between neutral, AXT-feature, and injected geometry providers."""

    def __init__(
        self,
        config: AxiomModelConfig,
        *,
        injected_provider: GeometryProviderProtocol | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.injected_provider = injected_provider
        self.null_provider = NullGeometryProvider(config.model_dim)
        self.axt_provider = AxtGeometryFeatureProvider(config)

    def forward(
        self,
        model_input: AxiomModelInput,
        slot_states: Tensor,
        relation_graph: object | None = None,
    ) -> GeometryContext:
        mode = self.config.effective_geometry_mode()
        if mode == "geometry_features_from_axt":
            return cast(
                GeometryContext,
                self.axt_provider(model_input, slot_states, relation_graph),
            )
        if mode == "geometry_provider_injected" and self.injected_provider is not None:
            context = self.injected_provider.forward(model_input, slot_states, relation_graph)
            diagnostics = {
                **context.diagnostics,
                "mode": "geometry_provider_injected",
                "gauge_invariant_only": True,
            }
            return GeometryContext(
                observables=context.observables,
                conditioning=context.conditioning,
                diagnostics=diagnostics,
                regularizer_terms=context.regularizer_terms,
            )
        return cast(GeometryContext, self.null_provider(model_input, slot_states, relation_graph))
