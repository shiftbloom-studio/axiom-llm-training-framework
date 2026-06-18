from __future__ import annotations

from pathlib import Path

from hcaps.experiments.configs import ExperimentSuiteConfig
from hcaps.experiments.orchestrator import ExperimentOrchestrator
from hcaps.training.config import ComputeBudgetConfig, CurriculumConfig, LossWeights, TrainingConfig

ROOT = Path(__file__).resolve().parents[2]


def test_experiment_suite_expands_arms_deterministically(
    p6_compiled_axt: Path,
    tmp_path: Path,
) -> None:
    template = TrainingConfig(
        run_name="p6_plan_test",
        seed=13,
        input_axt_path=p6_compiled_axt,
        output_dir=tmp_path,
        model_config_path=ROOT / "configs" / "model" / "structured_native_smoke.yaml",
        arm_name="D_structured_native_no_geometry",
        max_steps=1,
        max_epochs=1,
        global_batch_size=2,
        micro_batch_size=2,
        structured_loss_weights=LossWeights(),
        curriculum_config=CurriculumConfig(schedule_name="none"),
        compute_budget_config=ComputeBudgetConfig(max_train_steps=1),
        overwrite=True,
    )
    suite = ExperimentSuiteConfig(
        suite_name="p6_plan_test",
        input_axt=p6_compiled_axt,
        output_dir=tmp_path,
        arms=["A_flat_text", "D_structured_native_no_geometry"],
        seeds=[13],
        trainer_config_template=template,
        model_config_template=ROOT / "configs" / "model" / "structured_native_smoke.yaml",
    )

    plan = ExperimentOrchestrator(suite).prepare()

    assert [arm.arm_id for arm in plan.arms] == [
        "A_flat_text",
        "D_structured_native_no_geometry",
    ]
    assert plan.plan_path.exists()
