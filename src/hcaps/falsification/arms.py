"""Experiment arm declarations for falsification-first Axiom runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ExperimentArmName(StrEnum):
    """Supported deterministic comparison arms."""

    FLAT_TEXT = "flat_text"
    STRUCTURED_TEXT = "structured_text"
    CAPSULE_TEXT = "capsule_text"
    CAPSULE_NO_PROVENANCE = "capsule_no_provenance"
    CAPSULE_NO_RELATIONS = "capsule_no_relations"
    CAPSULE_NO_CONTEXT = "capsule_no_context"
    CAPSULE_NO_SIDE_CHANNELS = "capsule_no_side_channels"
    CONTEXT_SHUFFLE = "context_shuffle"
    PROVENANCE_SHUFFLE = "provenance_shuffle"
    POPULARITY_FREQUENCY_CONTROL = "popularity_frequency_control"
    TEMPORAL_HOLDOUT = "temporal_holdout"
    RELATION_ABLATION = "relation_ablation"


TEXT_RENDER_MODES: dict[ExperimentArmName, str] = {
    ExperimentArmName.FLAT_TEXT: "flat_text",
    ExperimentArmName.STRUCTURED_TEXT: "structured_text",
    ExperimentArmName.CAPSULE_TEXT: "capsule",
    ExperimentArmName.CAPSULE_NO_PROVENANCE: "capsule",
    ExperimentArmName.CAPSULE_NO_RELATIONS: "capsule",
    ExperimentArmName.CAPSULE_NO_CONTEXT: "capsule",
    ExperimentArmName.CAPSULE_NO_SIDE_CHANNELS: "capsule",
    ExperimentArmName.CONTEXT_SHUFFLE: "capsule",
    ExperimentArmName.PROVENANCE_SHUFFLE: "capsule",
    ExperimentArmName.POPULARITY_FREQUENCY_CONTROL: "structured_text",
    ExperimentArmName.TEMPORAL_HOLDOUT: "capsule",
    ExperimentArmName.RELATION_ABLATION: "capsule",
}


@dataclass(frozen=True)
class ExperimentArmConfig:
    """Declarative configuration for one generated arm."""

    name: ExperimentArmName
    description: str
    input_path: Path | None = None
    output_path: Path | None = None
    render_mode: str | None = None
    included_fields: tuple[str, ...] = field(default_factory=tuple)
    excluded_fields: tuple[str, ...] = field(default_factory=tuple)
    control_type: str = "baseline"
    deterministic_seed: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "description": self.description,
            "input_path": str(self.input_path) if self.input_path else None,
            "output_path": str(self.output_path) if self.output_path else None,
            "render_mode": self.render_mode,
            "included_fields": list(self.included_fields),
            "excluded_fields": list(self.excluded_fields),
            "control_type": self.control_type,
            "deterministic_seed": self.deterministic_seed,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ExperimentArmArtifact:
    """Paths and metadata produced for one arm."""

    name: ExperimentArmName
    output_dir: Path
    rendered_path: Path
    capsule_artifact_path: Path | None
    arm_manifest_path: Path
    record_count: int
    derived_noncanonical: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "output_dir": str(self.output_dir),
            "rendered_path": str(self.rendered_path),
            "capsule_artifact_path": str(self.capsule_artifact_path)
            if self.capsule_artifact_path
            else None,
            "arm_manifest_path": str(self.arm_manifest_path),
            "record_count": self.record_count,
            "derived_noncanonical": self.derived_noncanonical,
        }


def all_arm_names() -> list[ExperimentArmName]:
    """Return arms in stable manifest order."""

    return list(ExperimentArmName)


def parse_arm_names(values: list[str] | tuple[str, ...] | str) -> list[ExperimentArmName]:
    """Parse CLI/config arm names, supporting ``all``."""

    if isinstance(values, str):
        raw_values = [part.strip() for part in values.split(",") if part.strip()]
    else:
        raw_values = [part.strip() for part in values if part.strip()]
    if not raw_values or raw_values == ["all"] or "all" in raw_values:
        return all_arm_names()
    return [ExperimentArmName(value) for value in raw_values]


def default_arm_config(
    name: ExperimentArmName,
    *,
    input_path: Path | None = None,
    output_path: Path | None = None,
    seed: int | None = None,
) -> ExperimentArmConfig:
    """Build the default declaration for an arm."""

    descriptions = {
        ExperimentArmName.FLAT_TEXT: "Plain surface/canonical text baseline.",
        ExperimentArmName.STRUCTURED_TEXT: (
            "Stable headings with claim, context, and source labels."
        ),
        ExperimentArmName.CAPSULE_TEXT: (
            "Rich predictor-visible capsule text without future targets."
        ),
        ExperimentArmName.CAPSULE_NO_PROVENANCE: "Capsule text after provenance removal.",
        ExperimentArmName.CAPSULE_NO_RELATIONS: "Capsule text after relation removal.",
        ExperimentArmName.CAPSULE_NO_CONTEXT: "Capsule text after context removal.",
        ExperimentArmName.CAPSULE_NO_SIDE_CHANNELS: (
            "Capsule text with numeric side channels disabled."
        ),
        ExperimentArmName.CONTEXT_SHUFFLE: "Capsules with context assignments shuffled by seed.",
        ExperimentArmName.PROVENANCE_SHUFFLE: (
            "Capsules with provenance assignments shuffled by seed."
        ),
        ExperimentArmName.POPULARITY_FREQUENCY_CONTROL: (
            "Control retaining simple frequency and source-count proxies only."
        ),
        ExperimentArmName.TEMPORAL_HOLDOUT: "Holdout arm based on valid-as-of temporal cutoffs.",
        ExperimentArmName.RELATION_ABLATION: (
            "Capsules with relations and relation-derived signals removed."
        ),
    }
    included_fields = {
        ExperimentArmName.FLAT_TEXT: ("surface_forms", "claim"),
        ExperimentArmName.STRUCTURED_TEXT: ("claim", "context", "provenance"),
        ExperimentArmName.CAPSULE_TEXT: (
            "claim",
            "surface_forms",
            "epistemic_state",
            "provenance",
            "relations",
            "context",
        ),
        ExperimentArmName.CAPSULE_NO_PROVENANCE: (
            "claim",
            "surface_forms",
            "epistemic_state",
            "relations",
            "context",
        ),
        ExperimentArmName.CAPSULE_NO_RELATIONS: (
            "claim",
            "surface_forms",
            "epistemic_state",
            "provenance",
            "context",
        ),
        ExperimentArmName.CAPSULE_NO_CONTEXT: (
            "claim",
            "surface_forms",
            "epistemic_state",
            "provenance",
            "relations",
        ),
        ExperimentArmName.CAPSULE_NO_SIDE_CHANNELS: (
            "claim",
            "surface_forms",
            "epistemic_state",
            "provenance",
            "relations",
            "context",
        ),
        ExperimentArmName.CONTEXT_SHUFFLE: (
            "claim",
            "surface_forms",
            "epistemic_state",
            "provenance",
            "relations",
            "context",
        ),
        ExperimentArmName.PROVENANCE_SHUFFLE: (
            "claim",
            "surface_forms",
            "epistemic_state",
            "provenance",
            "relations",
            "context",
        ),
        ExperimentArmName.POPULARITY_FREQUENCY_CONTROL: (
            "claim",
            "surface_forms",
            "source_count_proxy",
            "claim_family_frequency_proxy",
        ),
        ExperimentArmName.TEMPORAL_HOLDOUT: (
            "claim",
            "surface_forms",
            "epistemic_state",
            "provenance",
            "relations",
            "context",
        ),
        ExperimentArmName.RELATION_ABLATION: (
            "claim",
            "surface_forms",
            "epistemic_state",
            "provenance",
            "context",
        ),
    }
    excluded_fields = {
        ExperimentArmName.FLAT_TEXT: ("epistemic_state", "provenance", "relations", "context"),
        ExperimentArmName.STRUCTURED_TEXT: ("training_targets", "quality", "lineage"),
        ExperimentArmName.CAPSULE_TEXT: ("training_targets",),
        ExperimentArmName.CAPSULE_NO_PROVENANCE: ("provenance", "training_targets"),
        ExperimentArmName.CAPSULE_NO_RELATIONS: ("relations", "training_targets"),
        ExperimentArmName.CAPSULE_NO_CONTEXT: ("context", "training_targets"),
        ExperimentArmName.CAPSULE_NO_SIDE_CHANNELS: ("side_channels", "training_targets"),
        ExperimentArmName.CONTEXT_SHUFFLE: ("training_targets",),
        ExperimentArmName.PROVENANCE_SHUFFLE: ("training_targets",),
        ExperimentArmName.POPULARITY_FREQUENCY_CONTROL: (
            "epistemic_state",
            "relations",
            "context",
            "training_targets",
            "provenance_details",
        ),
        ExperimentArmName.TEMPORAL_HOLDOUT: ("training_targets",),
        ExperimentArmName.RELATION_ABLATION: (
            "relations",
            "relation_side_channels",
            "training_targets",
        ),
    }
    control_types = {
        ExperimentArmName.FLAT_TEXT: "baseline",
        ExperimentArmName.STRUCTURED_TEXT: "baseline",
        ExperimentArmName.CAPSULE_TEXT: "treatment",
        ExperimentArmName.CAPSULE_NO_PROVENANCE: "ablation",
        ExperimentArmName.CAPSULE_NO_RELATIONS: "ablation",
        ExperimentArmName.CAPSULE_NO_CONTEXT: "ablation",
        ExperimentArmName.CAPSULE_NO_SIDE_CHANNELS: "ablation",
        ExperimentArmName.CONTEXT_SHUFFLE: "shuffle_control",
        ExperimentArmName.PROVENANCE_SHUFFLE: "shuffle_control",
        ExperimentArmName.POPULARITY_FREQUENCY_CONTROL: "frequency_control",
        ExperimentArmName.TEMPORAL_HOLDOUT: "temporal_control",
        ExperimentArmName.RELATION_ABLATION: "ablation",
    }
    return ExperimentArmConfig(
        name=name,
        description=descriptions[name],
        input_path=input_path,
        output_path=output_path,
        render_mode=TEXT_RENDER_MODES[name],
        included_fields=included_fields[name],
        excluded_fields=excluded_fields[name],
        control_type=control_types[name],
        deterministic_seed=seed
        if name
        in {
            ExperimentArmName.CONTEXT_SHUFFLE,
            ExperimentArmName.PROVENANCE_SHUFFLE,
            ExperimentArmName.POPULARITY_FREQUENCY_CONTROL,
            ExperimentArmName.TEMPORAL_HOLDOUT,
        }
        else None,
    )
