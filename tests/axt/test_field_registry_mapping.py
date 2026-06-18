from __future__ import annotations

from pathlib import Path

from hcaps.axt.registry import load_field_registry


def test_field_registry_mapping_resolves_required_fields() -> None:
    field_registry = Path(__file__).resolve().parents[2] / "spec" / "FIELD_REGISTRY_V1.md"
    registry = load_field_registry(field_registry)

    assert registry.resolve_tensor_group("claim.canonical_text") == "claim/text_projection"
    assert registry.resolve_visibility("future_target") == "target_only"
    assert registry.resolve_dtype("claim_type") == "int64"
    assert registry.resolve_mask_behavior("uncertainty") == "loss_mask_uncertainty_calibration"
