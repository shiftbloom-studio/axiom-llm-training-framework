from __future__ import annotations

from hcaps.axt import AxtBatch
from hcaps.model import AxiomModelConfig, AxiomStructuredModel
from tests.model._helpers import config_with


def test_null_geometry_provider_is_shape_compatible(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    output = AxiomStructuredModel(smoke_config)(axt_batch)
    assert output.geometry_diagnostics["enabled"] is False
    assert output.geometry_diagnostics["gauge_invariant_only"] is True
    assert output.geometry_diagnostics["raw_gauge_matrices_emitted"] is False


def test_geometry_feature_path_uses_axt_observables_without_raw_matrices(
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    config = config_with(
        smoke_config,
        use_geometry_features=True,
        geometry_mode="geometry_features_from_axt",
        ablations={"geometry_off": False},
    )
    output = AxiomStructuredModel(config)(axt_batch)
    assert output.geometry_diagnostics["enabled"] is True
    assert output.geometry_diagnostics["mode"] == "geometry_features_from_axt"
    assert output.geometry_diagnostics["raw_gauge_matrices_emitted"] is False
