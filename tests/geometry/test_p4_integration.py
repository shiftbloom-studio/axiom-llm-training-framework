from __future__ import annotations

from hcaps.geometry import GeometryConfig, P4GeometryProvider
from hcaps.model import AxiomModelConfig, AxiomStructuredModel


def test_p5_geometry_provider_plugs_into_p4_model(geometry_axt_batch) -> None:
    model_config = AxiomModelConfig.from_mapping(
        {
            "model": {"model_dim": 64, "slot_dim": 64, "num_layers": 2, "num_heads": 2},
            "inputs": {"text_vocab_size": 512, "categorical_vocab_size": 512},
            "geometry": {"enabled": True, "mode": "geometry_provider_injected"},
            "ablations": {"geometry_off": False},
        }
    )
    geometry_config = GeometryConfig(mode="learned", geometry_feature_dim=64)
    model = AxiomStructuredModel(
        model_config,
        geometry_provider=P4GeometryProvider(geometry_config, model_dim=64),
    )
    output = model(geometry_axt_batch)
    assert output.geometry_diagnostics["p5_geometry_provider"] is True
    assert output.geometry_diagnostics["gauge_invariant_only"] is True
    assert output.geometry_diagnostics["raw_gauge_matrices_emitted"] is False
