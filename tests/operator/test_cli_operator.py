from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from hcaps.cli import app
from hcaps.training.logging import write_json

ROOT = Path(__file__).resolve().parents[2]


def test_cli_p6_experiment_score_verdict_and_run_inspect(tmp_path: Path) -> None:
    runner = CliRunner()
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "suite_name": "p6_cli_smoke",
                "input_axp": str(ROOT / "examples" / "axf" / "v0_1" / "minimal_dataset.axp"),
                "input_axt": str(tmp_path / "minimal.axt"),
                "output_dir": str(tmp_path),
                "arms": ["A_flat_text", "D_structured_native_no_geometry"],
                "seeds": [13],
                "budget_profile": "smoke",
                "model_config_template": str(
                    ROOT / "configs" / "model" / "structured_native_smoke.yaml"
                ),
                "geometry_config_template": str(
                    ROOT / "configs" / "geometry" / "geometry_learned_smoke.yaml"
                ),
                "source_content_id": "cli_same_source",
                "split_id": "all_without_split",
                "trainer_config_template": {
                    "run_name": "p6_cli_smoke",
                    "seed": 13,
                    "input_axt_path": str(tmp_path / "minimal.axt"),
                    "output_dir": str(tmp_path),
                    "model_config_path": str(
                        ROOT / "configs" / "model" / "structured_native_smoke.yaml"
                    ),
                    "arm_name": "D_structured_native_no_geometry",
                    "max_steps": 1,
                    "max_epochs": 1,
                    "global_batch_size": 2,
                    "micro_batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "optimizer": "adamw",
                    "scheduler": "none",
                    "warmup_steps": 0,
                    "cooldown_steps": 0,
                    "clip_grad_norm": 1.0,
                    "precision": "fp32",
                    "checkpoint_interval": 1,
                    "eval_interval": 1,
                    "log_interval": 1,
                    "save_optimizer_state": True,
                    "resume_from": None,
                    "text_projection_loss_weight": 0.2,
                    "geometry_loss_weight": 0.0,
                    "geometry_config_path": None,
                    "control_transform": None,
                    "text_mode": None,
                    "overwrite": True,
                    "structured_loss_weights": {
                        "structured_axc_out": 1.0,
                        "relation_prediction": 1.0,
                        "provenance_recovery": 1.0,
                        "epistemic_proxy": 1.0,
                        "stability_temporal": 1.0,
                        "uncertainty_calibration": 0.5,
                        "geometry_observables": 0.0,
                        "geometry_regularization": 0.0,
                        "text_projection": 0.2,
                    },
                    "curriculum_config": {
                        "schedule_name": "none",
                        "phase_boundaries": [0, 1],
                        "text_projection_weight_schedule": [1.0],
                        "structured_loss_weight_schedule": [1.0],
                        "relation_neighborhood_depth_schedule": [1],
                        "side_channel_dropout_schedule": [0.0],
                        "geometry_activation_schedule": [0.0],
                        "context_dropout_schedule": [0.0],
                        "negative_sample_hardness_schedule": [0.0],
                    },
                    "compute_budget_config": {
                        "max_train_steps": 1,
                        "max_records_seen": 8,
                        "max_wall_clock_seconds": 120.0,
                        "parameter_match_tolerance": 0.25,
                        "compute_match_tolerance": 0.25,
                        "estimate_flops": True,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    run = runner.invoke(app, ["experiment", "run", str(suite_path), "--json"])
    assert run.exit_code == 0, run.output
    run_dir = tmp_path / "p6_cli_smoke"

    score = runner.invoke(app, ["score", "run", str(run_dir), "--json"])
    assert score.exit_code == 0, score.output

    report = runner.invoke(app, ["verdict", "report", str(run_dir), "--json"])
    assert report.exit_code == 0, report.output

    inspect = runner.invoke(app, ["run", "inspect", str(run_dir), "--json"])
    assert inspect.exit_code == 0, inspect.output


def test_operator_export_command(tmp_path: Path) -> None:
    runner = CliRunner()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "run_manifest.json", {"run_id": "run", "arms": {}})

    result = runner.invoke(
        app,
        ["run", "export", str(run_dir), "--output", str(tmp_path / "bundle.json"), "--json"],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "bundle.json").exists()
