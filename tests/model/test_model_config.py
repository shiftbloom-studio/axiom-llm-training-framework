from __future__ import annotations

import pytest

from hcaps.model import AxiomModelConfig, AxiomStructuredModel
from hcaps.model.parameter_count import parameter_count


def test_smoke_config_loads_and_reports_budget(smoke_config: AxiomModelConfig) -> None:
    model = AxiomStructuredModel(smoke_config)
    counts = parameter_count(model)
    assert smoke_config.parameter_budget_hint == "smoke"
    assert 500_000 <= counts["total"] <= 2_000_000
    assert smoke_config.effective_text_projection()


def test_invalid_dimensions_fail_clearly(smoke_config: AxiomModelConfig) -> None:
    payload = smoke_config.model_dump(mode="python")
    payload["model_dim"] = 63
    payload["num_heads"] = 8
    with pytest.raises(ValueError, match="divisible"):
        AxiomModelConfig.model_validate(payload)


def test_ablation_and_geometry_modes_validate(smoke_config: AxiomModelConfig) -> None:
    payload = smoke_config.model_dump(mode="python")
    payload["geometry_mode"] = "geometry_features_from_axt"
    payload["use_geometry_features"] = True
    payload["ablations"]["geometry_off"] = False
    payload["ablations"]["router_off"] = True
    config = AxiomModelConfig.model_validate(payload)
    assert config.effective_geometry_mode() == "geometry_features_from_axt"
    assert config.effective_router_mode() == "router_off"
