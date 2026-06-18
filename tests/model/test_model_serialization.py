from __future__ import annotations

from pathlib import Path

from hcaps.axt import AxtBatch
from hcaps.model import AxiomModelConfig, AxiomStructuredModel
from hcaps.model.serialization import load_model_checkpoint, save_model_checkpoint


def test_model_checkpoint_roundtrip_preserves_forward_shapes(
    tmp_path: Path,
    smoke_config: AxiomModelConfig,
    axt_batch: AxtBatch,
) -> None:
    model = AxiomStructuredModel(smoke_config)
    original = model(axt_batch)
    checkpoint_path = tmp_path / "model.pt"
    save_model_checkpoint(checkpoint_path, model, smoke_config, {"purpose": "p4_test"})

    loaded_model, loaded_config, metadata = load_model_checkpoint(checkpoint_path)
    loaded = loaded_model(axt_batch)

    assert loaded_config.p4_model_schema_version == smoke_config.p4_model_schema_version
    assert metadata["compatible_axt_spec_version"] == smoke_config.compatible_axt_spec_version
    assert metadata["compatible_axc_out_spec_version"] == (
        smoke_config.compatible_axc_out_spec_version
    )
    assert loaded.latent_state.shape == original.latent_state.shape
    assert loaded.raw_axc_out.head_shapes() == original.raw_axc_out.head_shapes()
