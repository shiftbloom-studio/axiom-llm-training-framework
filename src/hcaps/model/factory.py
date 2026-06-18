"""Factory and main PyTorch module for the P4 structured-native stack."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from torch import Tensor, nn

from hcaps.model.config import AxiomModelConfig
from hcaps.model.core import FullComplexityCore
from hcaps.model.decoder import StructuredDecoder
from hcaps.model.diagnostics import model_diagnostics
from hcaps.model.embeddings import TypedFieldEmbeddingSet
from hcaps.model.encoder import StructuredEncoder
from hcaps.model.geometry_hooks import GeometryHook, GeometryProviderProtocol
from hcaps.model.input_adapter import StructuredInputAdapter
from hcaps.model.interpreter import AXCOutInterpreterBoundary
from hcaps.model.provenance_conditioning import ProvenanceConditioner
from hcaps.model.relation_conditioning import RelationNeighborhoodConditioner
from hcaps.model.router import EpistemicRouter
from hcaps.model.text_projection import TextProjectionHead
from hcaps.model.types import AxiomModelInput, AxiomModelOutput


class AxiomStructuredModel(nn.Module):
    """Structured-native LLM model stack for AXT -> AXC-out + text projection."""

    def __init__(
        self,
        config: AxiomModelConfig,
        *,
        geometry_provider: GeometryProviderProtocol | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_adapter = StructuredInputAdapter(config)
        self.embeddings = TypedFieldEmbeddingSet(config)
        self.encoder = StructuredEncoder(config)
        self.relation_conditioner = RelationNeighborhoodConditioner(config)
        self.provenance_conditioner = ProvenanceConditioner(config)
        self.geometry_hook = GeometryHook(config, injected_provider=geometry_provider)
        self.core = FullComplexityCore(config)
        self.router = EpistemicRouter(config)
        self.decoder = StructuredDecoder(config)
        self.interpreter = AXCOutInterpreterBoundary()
        self.text_projection_head = TextProjectionHead(config)

    def forward(self, batch: object) -> AxiomModelOutput:
        model_input = batch if isinstance(batch, AxiomModelInput) else self.input_adapter(batch)
        slots = self.embeddings(model_input)
        encoded = self.encoder(slots)

        relation_output = self.relation_conditioner(
            encoded.slot_states,
            encoded.slot_mask,
            model_input,
        )
        encoded = replace(encoded, slot_states=relation_output.slot_states)
        provenance_output = self.provenance_conditioner(
            encoded.slot_states,
            encoded.slot_mask,
            model_input,
        )
        encoded = replace(encoded, slot_states=provenance_output.slot_states)

        geometry_context = self.geometry_hook(model_input, encoded.slot_states, relation_graph=None)
        core_output = self.core(
            encoded,
            model_input,
            relation_state=relation_output.relation_state,
            provenance_state=provenance_output.provenance_state,
            geometry_context=geometry_context,
        )
        router_output = self.router(core_output, model_input, geometry_context)
        raw_emission = self.decoder(router_output.latent_state, model_input)

        text_logits = self._text_projection_logits(router_output.latent_state, model_input)
        axc_out = self.interpreter(
            raw_emission,
            text_projection_shape=tuple(text_logits.shape) if text_logits is not None else None,
        )
        diagnostics = model_diagnostics(
            config=self.config,
            model=self,
            model_input=model_input,
            slots=slots,
            axc_out=axc_out,
            text_projection_logits=text_logits,
            router_summary=router_output.diagnostics,
            geometry_summary=geometry_context.diagnostics,
            stage_diagnostics={
                "embedding": slots.diagnostics,
                "encoder": encoded.diagnostics,
                "relation_conditioning": relation_output.diagnostics,
                "provenance_conditioning": provenance_output.diagnostics,
                "core": core_output.diagnostics,
            },
        )
        return AxiomModelOutput(
            latent_state=router_output.latent_state,
            slot_states=router_output.slot_states,
            raw_axc_out=axc_out,
            text_projection_logits=text_logits,
            auxiliary_logits={
                "head_weights": router_output.head_weights,
                **router_output.loss_term_weights,
                **{
                    f"geometry_regularizer_{name}": value
                    for name, value in (geometry_context.regularizer_terms or {}).items()
                },
            },
            router_diagnostics=router_output.diagnostics,
            geometry_diagnostics=geometry_context.diagnostics,
            masks={
                "loss_masks": model_input.loss_masks,
                "availability_masks": model_input.availability_masks,
            },
            metadata={
                **model_input.metadata,
                "p4_model_schema_version": self.config.p4_model_schema_version,
            },
            diagnostics=diagnostics,
        )

    def _text_projection_logits(
        self,
        latent_state: Tensor,
        model_input: AxiomModelInput,
    ) -> Tensor | None:
        if not self.config.effective_text_projection():
            return None
        text_group = model_input.group("text_projection")
        text_ids = text_group.get("text_input_ids")
        text_mask = text_group.get("text_attention_mask")
        return cast(
            Tensor,
            self.text_projection_head(
                latent_state,
                text_input_ids=text_ids,
                text_attention_mask=text_mask,
            ),
        )
