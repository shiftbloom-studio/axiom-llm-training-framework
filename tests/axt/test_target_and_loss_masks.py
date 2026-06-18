from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle


def test_target_and_loss_masks_do_not_treat_missing_targets_as_negatives(
    compiled_minimal_axp: Path,
) -> None:
    bundle = AxtBundle(compiled_minimal_axp)
    availability = bundle.read_tensor_group("availability_masks")
    loss = bundle.read_tensor_group("loss_masks")
    targets = bundle.read_tensor_group("targets")

    assert availability["available_future_targets"].tolist() == [0, 0]
    assert loss["loss_mask_future_summary"].tolist() == [0, 0]
    assert targets["future_summary_target_ref"].tolist() == [-1, -1]
    assert "missing_target_is_not_negative" in (
        bundle.path / "reports" / "missing_target_report.json"
    ).read_text(encoding="utf-8")
