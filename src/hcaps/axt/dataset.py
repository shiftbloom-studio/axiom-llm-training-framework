"""Runtime dataset interface for AXT bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .reader import AxtBundle


@dataclass(frozen=True)
class AxtRecord:
    """One structured runtime item from an AXT bundle."""

    index: int
    ids: dict[str, Any]
    claim: dict[str, Any]
    text_projection: dict[str, Any]
    temporal: dict[str, Any]
    lateral_context: dict[str, Any]
    provider_context: dict[str, Any]
    epistemic: dict[str, Any]
    provenance: dict[str, Any]
    relations: dict[str, Any]
    neighborhood: dict[str, Any]
    negative_samples: dict[str, Any]
    geometry_slots: dict[str, Any]
    targets: dict[str, Any]
    availability_masks: dict[str, Any]
    loss_masks: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ids": self.ids,
            "claim": self.claim,
            "text_projection": self.text_projection,
            "temporal": self.temporal,
            "lateral_context": self.lateral_context,
            "provider_context": self.provider_context,
            "epistemic": self.epistemic,
            "provenance": self.provenance,
            "relations": self.relations,
            "neighborhood": self.neighborhood,
            "negative_samples": self.negative_samples,
            "geometry_slots": self.geometry_slots,
            "targets": self.targets,
            "availability_masks": self.availability_masks,
            "loss_masks": self.loss_masks,
            "metadata": self.metadata,
        }


class AxtDataset:
    """Load AXT tensor groups and expose structured runtime records."""

    def __init__(self, path: str | Path, *, split_name: str | None = None) -> None:
        self.bundle = AxtBundle(path)
        self.indices = self.bundle.load_selected_split(split_name)
        self._groups = {
            group_name: self.bundle.read_tensor_group(group_name)
            for group_name in self.bundle.tensor_group_names()
        }
        self._source_index = self.bundle.source_index()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> AxtRecord:
        index = self.indices[item]
        return AxtRecord(
            index=index,
            ids=self._dense("ids", index),
            claim=self._dense("claim", index),
            text_projection=self._dense("text_projection", index),
            temporal=self._dense("temporal", index),
            lateral_context=self._dense("lateral_context", index),
            provider_context=self._dense("provider_context", index),
            epistemic=self._dense("epistemic_state", index),
            provenance=self._ragged("provenance", index),
            relations=self._ragged("relations", index),
            neighborhood=self._ragged("relation_neighborhoods", index),
            negative_samples=self._ragged("negative_samples", index),
            geometry_slots=self._dense("geometry_observables", index),
            targets=self._ragged("targets", index),
            availability_masks=self._dense("availability_masks", index),
            loss_masks=self._dense("loss_masks", index),
            metadata={
                "record_index": index,
                "source_index": self._record_source_index(index),
            },
        )

    def _dense(self, group_name: str, index: int) -> dict[str, Any]:
        group = self._groups[group_name]
        result: dict[str, Any] = {}
        for name, array in group.items():
            if array.ndim > 0 and array.shape[0] == self.bundle.manifest.record_count:
                result[name] = _to_jsonable(array[index])
            else:
                result[name] = _to_jsonable(array)
        return result

    def _ragged(self, group_name: str, index: int) -> dict[str, Any]:
        group = self._groups[group_name]
        result = self._dense(group_name, index)
        for offsets_name in (name for name in group if name.endswith("offsets")):
            stem = offsets_name[: -len("offsets")].rstrip("_")
            offsets = group[offsets_name]
            if offsets.ndim != 1 or len(offsets) <= index + 1:
                continue
            start = int(offsets[index])
            end = int(offsets[index + 1])
            for value_name, values in group.items():
                if value_name == offsets_name or value_name.endswith("offsets"):
                    continue
                if values.ndim == 1 and values.shape[0] >= end and value_name.startswith(stem):
                    result[f"{value_name}_slice"] = _to_jsonable(values[start:end])
        return result

    def _record_source_index(self, index: int) -> dict[str, Any]:
        record_ids = self._source_index.get("records", [])
        if isinstance(record_ids, list) and index < len(record_ids):
            record = record_ids[index]
            return record if isinstance(record, dict) else {}
        return {}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value
