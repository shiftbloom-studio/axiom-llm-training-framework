from __future__ import annotations

from pathlib import Path

from hcaps.training import AxiomTrainer, TrainingConfig
from hcaps.training.config import ComputeBudgetConfig, CurriculumConfig, LossWeights

ROOT = Path(__file__).resolve().parents[2]


def _training_config(tmp_path: Path, axt_path: Path, *, max_steps: int = 1) -> TrainingConfig:
    return TrainingConfig(
        run_name="p6_trainer_test",
        seed=13,
        input_axt_path=axt_path,
        output_dir=tmp_path,
        model_config_path=ROOT / "configs" / "model" / "structured_native_smoke.yaml",
        arm_name="D_structured_native_no_geometry",
        max_steps=max_steps,
        max_epochs=1,
        global_batch_size=2,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        checkpoint_interval=1,
        eval_interval=1,
        log_interval=1,
        structured_loss_weights=LossWeights(geometry_observables=0.0, geometry_regularization=0.0),
        text_projection_loss_weight=0.2,
        geometry_loss_weight=0.0,
        curriculum_config=CurriculumConfig(schedule_name="none"),
        compute_budget_config=ComputeBudgetConfig(max_train_steps=max_steps),
        overwrite=True,
    )


def test_trainer_runs_one_step_and_writes_checkpoint(p6_compiled_axt: Path, tmp_path: Path) -> None:
    result = AxiomTrainer(_training_config(tmp_path, p6_compiled_axt)).fit()

    assert result.final_step == 1
    assert result.checkpoint_path.exists()
    assert result.metrics_path.exists()
    assert (result.arm_dir / "predictions" / "raw_emission.jsonl").exists()


def test_trainer_resume_preserves_step_and_rng_state(
    p6_compiled_axt: Path,
    tmp_path: Path,
) -> None:
    first = AxiomTrainer(_training_config(tmp_path, p6_compiled_axt, max_steps=1)).fit()
    resumed_config = _training_config(tmp_path, p6_compiled_axt, max_steps=2).model_copy(
        update={"resume_from": first.checkpoint_path}
    )
    resumed = AxiomTrainer(resumed_config).fit()

    assert resumed.final_step == 2
    assert (resumed.arm_dir / "checkpoints" / "latest.pt").exists()
