"""AXP v0.1 package manifest schema."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from hcaps.format.identifiers import PackageId


class AxfSchemaVersions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    axf: Literal["0.1.0"] = "0.1.0"
    axc: Literal["0.1.0"] = "0.1.0"
    axp: Literal["0.1.0"] = "0.1.0"
    axt: Literal["0.1.0"] = "0.1.0"


class AxpFileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    role: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    record_count: int | None = Field(default=None, ge=0)


class AxpCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capsules: int = Field(default=0, ge=0)
    sources: int = Field(default=0, ge=0)
    relations: int = Field(default=0, ge=0)
    contexts: int = Field(default=0, ge=0)


class AxpTemporalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cutoff_policy: str = "predictor_side_sources_must_not_exceed_valid_as_of"
    leakage_report: str = "manifests/leakage_report.json"
    temporal_holdout: str = "splits/temporal_holdout.json"


class AxpConstruction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    builder: str = Field(min_length=1)
    builder_version: str = Field(min_length=1)
    source_manifest: str | None = None
    construction_report: str = "manifests/construction_report.json"


class AxpManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    format: Literal["AXP"] = "AXP"
    format_version: Literal["0.1.0"] = "0.1.0"
    package_id: PackageId
    dataset_name: str = Field(min_length=1)
    created_at: AwareDatetime
    license: str = Field(min_length=1)
    schema_versions: AxfSchemaVersions = Field(default_factory=AxfSchemaVersions, alias="schema")
    files: list[AxpFileEntry] = Field(default_factory=list)
    counts: AxpCounts = Field(default_factory=AxpCounts)
    temporal_policy: AxpTemporalPolicy = Field(default_factory=AxpTemporalPolicy)
    construction: AxpConstruction
    hashes: dict[str, str] = Field(default_factory=dict)
    controls: dict[str, str] = Field(
        default_factory=lambda: {
            "temporal_holdouts": "splits/temporal_holdout.json",
            "context_shuffle": "manifests/confound_controls.json",
            "degree_preserving_rewires": "manifests/confound_controls.json",
            "popularity_recency_controls": "manifests/confound_controls.json",
            "embedding_only_baselines": "manifests/confound_controls.json",
            "relation_ablations": "manifests/confound_controls.json",
            "geometry_disabled_ablations": "manifests/confound_controls.json",
            "provenance_disabled_ablations": "manifests/confound_controls.json",
        }
    )
