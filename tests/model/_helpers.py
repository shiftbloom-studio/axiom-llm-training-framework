from __future__ import annotations

from hcaps.model import AxiomModelConfig


def config_with(config: AxiomModelConfig, **updates: object) -> AxiomModelConfig:
    payload = config.model_dump(mode="python")
    for key, value in updates.items():
        if key == "ablations" and isinstance(value, dict):
            payload["ablations"] = {**payload["ablations"], **value}
        else:
            payload[key] = value
    return AxiomModelConfig.model_validate(payload)
