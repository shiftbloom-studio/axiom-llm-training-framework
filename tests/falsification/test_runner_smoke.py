from __future__ import annotations

from pathlib import Path

from hcaps.falsification.runner import FalsificationRunConfig, run_falsification_harness

EXAMPLES = Path("examples/axf/v0_1")


def test_runner_smoke_writes_artifacts_and_manifest(tmp_path: Path) -> None:
    result = run_falsification_harness(
        FalsificationRunConfig(
            input_path=EXAMPLES / "minimal_dataset.axp",
            output_dir=tmp_path,
            arms=["flat_text", "structured_text", "capsule_text"],
            seed=13,
            allow_all_without_split=True,
        )
    )

    assert result.manifest_path.exists()
    assert (result.run_dir / "audits" / "input_audit.json").exists()
    assert (result.run_dir / "metrics" / "arm_metrics.json").exists()
    assert {arm.name.value for arm in result.arm_artifacts} == {
        "flat_text",
        "structured_text",
        "capsule_text",
    }
    for arm in result.arm_artifacts:
        assert arm.rendered_path.exists()
        assert arm.arm_manifest_path.exists()
