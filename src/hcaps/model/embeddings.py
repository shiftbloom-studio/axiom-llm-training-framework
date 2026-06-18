"""Typed field embeddings for structured AXT groups."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hcaps.model.config import AxiomModelConfig
from hcaps.model.features import group_mask, group_to_feature_tensor
from hcaps.model.types import SLOT_TYPE_IDS, AxiomModelInput, TypedSlotBundle


@dataclass(frozen=True)
class FieldSlot:
    tensor: Tensor
    type_id: int
    mask: Tensor
    source_group: str


class TypedFieldEmbedding(nn.Module):
    """Base module that turns one or more structured groups into one typed slot."""

    def __init__(
        self,
        config: AxiomModelConfig,
        *,
        slot_name: str,
        source_groups: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.config = config
        self.slot_name = slot_name
        self.source_groups = source_groups
        self.type_id = SLOT_TYPE_IDS[slot_name]
        self.numeric_projection = nn.Linear(config.numeric_feature_count, config.slot_dim)
        self.categorical_embedding = nn.Embedding(config.categorical_vocab_size, config.slot_dim)
        self.norm = nn.LayerNorm(config.slot_dim)

    def forward(self, model_input: AxiomModelInput) -> FieldSlot:
        features = torch.zeros(
            (model_input.batch_size, self.config.numeric_feature_count),
            dtype=torch.float32,
            device=model_input.device,
        )
        mask = torch.zeros((model_input.batch_size,), dtype=torch.bool, device=model_input.device)
        categorical_seed = torch.zeros(
            (model_input.batch_size,),
            dtype=torch.long,
            device=model_input.device,
        )
        for group_name in self.source_groups:
            group = model_input.group(group_name)
            features = features + group_to_feature_tensor(
                group,
                batch_size=model_input.batch_size,
                feature_count=self.config.numeric_feature_count,
                device=model_input.device,
            )
            mask |= group_mask(group, batch_size=model_input.batch_size, device=model_input.device)
            categorical_seed = categorical_seed + _categorical_seed(group, model_input.batch_size)
        categorical_seed = torch.remainder(
            torch.abs(categorical_seed),
            self.config.categorical_vocab_size,
        )
        slot_tensor = self.numeric_projection(features) + self.categorical_embedding(
            categorical_seed
        )
        return FieldSlot(
            tensor=self.norm(slot_tensor),
            type_id=self.type_id,
            mask=mask,
            source_group=self.slot_name,
        )


class ClaimIdentityEmbedding(TypedFieldEmbedding):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__(
            config,
            slot_name="claim_identity",
            source_groups=("ids", "claim"),
        )


class ClaimStateEmbedding(TypedFieldEmbedding):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__(config, slot_name="claim_state", source_groups=("claim",))


class EpistemicStateEmbedding(TypedFieldEmbedding):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__(config, slot_name="epistemic_state", source_groups=("epistemic_state",))


class TemporalEmbedding(TypedFieldEmbedding):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__(config, slot_name="temporal_scope", source_groups=("temporal",))


class LateralContextEmbedding(TypedFieldEmbedding):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__(config, slot_name="lateral_context", source_groups=("lateral_context",))


class ProviderContextEmbedding(TypedFieldEmbedding):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__(config, slot_name="provider_context", source_groups=("provider_context",))


class ProvenanceEmbedding(TypedFieldEmbedding):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__(config, slot_name="provenance", source_groups=("provenance",))


class RelationEmbedding(TypedFieldEmbedding):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__(config, slot_name="relations", source_groups=("relations",))


class RelationNeighborhoodEmbedding(TypedFieldEmbedding):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__(
            config,
            slot_name="relation_neighborhoods",
            source_groups=("relation_neighborhoods",),
        )


class NegativeSampleEmbedding(TypedFieldEmbedding):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__(config, slot_name="negative_samples", source_groups=("negative_samples",))


class GeometryFeatureEmbedding(TypedFieldEmbedding):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__(
            config,
            slot_name="geometry_features",
            source_groups=("geometry_observables",),
        )


class TextProjectionInputEmbedding(TypedFieldEmbedding):
    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__(
            config,
            slot_name="text_projection",
            source_groups=("text_projection",),
        )


class TypedFieldEmbeddingSet(nn.Module):
    """Compose required typed field embeddings and preserve slot metadata."""

    def __init__(self, config: AxiomModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleDict(
            {
                "claim_identity": ClaimIdentityEmbedding(config),
                "claim_state": ClaimStateEmbedding(config),
                "epistemic_state": EpistemicStateEmbedding(config),
                "temporal_scope": TemporalEmbedding(config),
                "lateral_context": LateralContextEmbedding(config),
                "provider_context": ProviderContextEmbedding(config),
                "provenance": ProvenanceEmbedding(config),
                "relations": RelationEmbedding(config),
                "relation_neighborhoods": RelationNeighborhoodEmbedding(config),
                "negative_samples": NegativeSampleEmbedding(config),
                "geometry_features": GeometryFeatureEmbedding(config),
                "text_projection": TextProjectionInputEmbedding(config),
            }
        )

    def forward(self, model_input: AxiomModelInput) -> TypedSlotBundle:
        active_names = self._active_embedding_names()
        slots = [self.embeddings[name](model_input) for name in active_names]
        if not slots:
            raise RuntimeError("at least one typed slot embedding must remain active")

        slot_tensor = torch.stack([slot.tensor for slot in slots], dim=1)
        slot_type_ids = torch.stack(
            [
                torch.full(
                    (model_input.batch_size,),
                    slot.type_id,
                    dtype=torch.long,
                    device=model_input.device,
                )
                for slot in slots
            ],
            dim=1,
        )
        slot_mask = torch.stack([slot.mask for slot in slots], dim=1)
        source_groups = tuple(slot.source_group for slot in slots)
        group_ranges = {name: (index, index + 1) for index, name in enumerate(source_groups)}
        return TypedSlotBundle(
            slot_tensor=slot_tensor,
            slot_type_ids=slot_type_ids,
            slot_mask=slot_mask,
            source_groups=source_groups,
            group_ranges=group_ranges,
            diagnostics={
                "slot_count": len(source_groups),
                "source_groups": source_groups,
                "active_ablation_modes": self.config.ablations.active,
            },
        )

    def _active_embedding_names(self) -> tuple[str, ...]:
        ablations = self.config.ablations
        if ablations.text_only:
            return ("text_projection",)

        names: list[str] = ["claim_identity", "claim_state", "epistemic_state"]
        if not ablations.no_side_channels:
            names.append("temporal_scope")
        if not ablations.no_context and not ablations.no_side_channels:
            names.append("lateral_context")
        if (
            self.config.use_provider_context
            and not ablations.no_provider_context
            and not ablations.no_side_channels
        ):
            names.append("provider_context")
        if not ablations.no_provenance and not ablations.no_side_channels:
            names.append("provenance")
        if not ablations.no_relations:
            names.append("relations")
            if not ablations.relation_neighborhood_off:
                names.append("relation_neighborhoods")
            names.append("negative_samples")
        if self.config.effective_geometry_mode() != "geometry_off":
            names.append("geometry_features")
        if self.config.effective_text_projection():
            names.append("text_projection")
        return tuple(names)


def _categorical_seed(group: dict[str, Tensor], batch_size: int) -> Tensor:
    device = next(iter(group.values())).device if group else torch.device("cpu")
    seed = torch.zeros((batch_size,), dtype=torch.long, device=device)
    for key in sorted(group):
        tensor = group[key]
        if tensor.dtype.is_floating_point:
            continue
        if tensor.ndim == 0:
            values = tensor.reshape(1).expand(batch_size)
        elif tensor.ndim == 1:
            values = tensor if tensor.shape[0] == batch_size else tensor[:1].expand(batch_size)
        else:
            values = tensor.reshape(tensor.shape[0], -1)[:, 0]
        seed = seed + values.to(dtype=torch.long)
    return seed
