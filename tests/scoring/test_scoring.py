from __future__ import annotations

from pathlib import Path

from hcaps.scoring import score_run
from hcaps.training.logging import write_json


def test_scoring_distinguishes_axc_out_levels(tmp_path: Path) -> None:
    arm_dir = tmp_path / "arms" / "D_structured_native_no_geometry"
    arm_dir.mkdir(parents=True)
    write_json(
        arm_dir / "metrics.json",
        {
            "validation": {
                "metrics": {
                    "loss_relation_prediction": 0.2,
                    "loss_provenance_recovery": 0.3,
                    "loss_epistemic_proxy": 0.1,
                    "text_projection_text_cross_entropy": 1.0,
                    "text_projection_text_perplexity": 2.7,
                    "text_projection_token_accuracy": 0.5,
                }
            }
        },
    )

    result = score_run(tmp_path)

    assert "raw_emission" in result.scores["levels_scored"]
    assert result.scores["arms"]["D_structured_native_no_geometry"]["raw_emission_scored"] is True
    assert result.scores_path.exists()
