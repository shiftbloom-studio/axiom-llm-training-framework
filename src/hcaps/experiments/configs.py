"""Typed experiment suite configuration for P6."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

from hcaps.training.config import TrainingConfig


class ExperimentSuiteConfig(BaseModel):
    """P6 multi-arm experiment suite configuration."""

    model_config = ConfigDict(extra="forbid")

    suite_name: str
    input_axp: Path | None = None
    input_axt: Path
    output_dir: Path
    arms: list[str] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=lambda: [13])
    budget_profile: str = "smoke"
    trainer_config_template: TrainingConfig
    model_config_template: Path
    geometry_config_template: Path | None = None
    scoring_config: Path | None = None
    verdict_config: Path | None = None
    source_content_id: str = "same_source_content"
    split_id: str = "all_without_split"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"experiment suite config must be a YAML mapping: {path}")
        return cls.model_validate(payload)

    def to_yaml(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True), "utf-8")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
