from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import yaml
from typer.testing import CliRunner

from hcaps.cli import _execute_full_pipeline, _full_p6_arm_ids, app
from hcaps.experiments.arms import default_arm_catalog
from hcaps.providers.config import ProviderIngressConfig
from hcaps.providers.llama_server_runtime import LlamaServerConfig
from hcaps.training.config import TrainingConfig
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

    compare = runner.invoke(app, ["experiment", "compare", str(run_dir), "--json"])
    assert compare.exit_code == 0, compare.output
    assert (run_dir / "comparisons" / "pairwise_metrics.json").exists()
    assert (run_dir / "comparisons" / "control_effects.json").exists()

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


def test_interactive_pipeline_execute_minimal(tmp_path: Path) -> None:
    """Direct test of the consolidated pipeline execute path (corpus->axt->experiment).

    Uses deterministic (no provider key), reduced arms, tiny budget to stay fast while
    exercising the unreduced code paths (full tensor groups in AXT, loss masks, etc).
    The interactive prompt wrapper is not exercised here; CLI bare-run is covered via
    monkeypatch in integration if needed.
    """
    # tiny trainer override based on existing test patterns (full fidelity defaults exercised
    # in the unreduced path; here we use a smoke-scale tiny variant for test runtime)
    tiny_trainer = TrainingConfig.model_validate(
        {
            "run_name": "placeholder",
            "seed": 13,
            "input_axt_path": str(tmp_path / "ph.axt"),
            "output_dir": str(tmp_path),
            "model_config_path": str(ROOT / "configs" / "model" / "structured_native_smoke.yaml"),
            "arm_name": "placeholder",
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
        }
    )

    sources = ROOT / "examples" / "corpus" / "ml_software_benchmarks" / "sources"
    run_name = "test_interactive_min"
    final = _execute_full_pipeline(
        run_name=run_name,
        sources_dir=sources,
        cutoff=date(2026, 1, 1),
        providers=None,
        arms=["A_flat_text", "D_structured_native_no_geometry"],
        trainer_template=tiny_trainer,
        output_root=tmp_path,
    )

    assert final.exists()
    assert (final / "axt").exists()
    assert (final / "corpus" / "dataset.axp").exists()
    assert (final / "wizard_configs" / "corpus.yaml").exists()
    assert (final / "wizard_configs" / "suite.yaml").exists()
    # at least one arm trained with checkpoint
    arm_dir = final / "arms" / "A_flat_text"
    assert arm_dir.exists()
    assert (arm_dir / "checkpoints" / "latest.pt").exists() or any(
        (arm_dir / "checkpoints").glob("step_*.pt")
    )
    # verdict artifacts from full flow
    assert (final / "verdict" / "report.md").exists() or (final / "verdict").exists()


def test_cli_bare_run_launches_wizard(monkeypatch, tmp_path: Path) -> None:
    """Verify that bare `axiom run` (no subcommand) triggers the interactive wizard
    path and collects prompts before delegating to execute.
    """
    called: dict[str, object] = {}

    def fake_execute(**kw: object) -> Path:
        called["run"] = kw.get("run_name")
        rd = tmp_path / str(kw.get("run_name", "x"))
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "verdict").mkdir(exist_ok=True)
        (rd / "verdict" / "report.md").write_text("# ok\n")
        return rd

    monkeypatch.setattr("hcaps.cli._execute_full_pipeline", fake_execute)

    runner = CliRunner()
    # Feed: accept default sources (\n), default cutoff (\n), no-provider (n),
    # run name, confirm ready (y). Matches prompt order in _run_interactive...
    inp = "\n\nn\ntest_bare_wizard\ny\n"
    res = runner.invoke(app, ["run"], input=inp)
    assert res.exit_code == 0, res.output
    assert "test_bare_wizard" in (res.output or "") or "Pipeline complete" in (res.output or "")
    assert called.get("run") == "test_bare_wizard"


def test_cli_bare_run_records_remote_provider_mode(monkeypatch, tmp_path: Path) -> None:
    called: dict[str, object] = {}

    def fake_execute(**kw: object) -> Path:
        called.update(kw)
        rd = tmp_path / str(kw.get("run_name", "x"))
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "verdict").mkdir(exist_ok=True)
        (rd / "verdict" / "report.md").write_text("# ok\n", encoding="utf-8")
        return rd

    monkeypatch.setattr("hcaps.cli._execute_full_pipeline", fake_execute)

    runner = CliRunner()
    inp = "\n\ny\nhttps://example.test/v1\nremote\nremote-model\nsecret-key\n\ntest_remote\ny\n"
    res = runner.invoke(app, ["run"], input=inp)

    assert res.exit_code == 0, res.output
    providers = called["providers"]
    assert isinstance(providers, ProviderIngressConfig)
    assert providers.primary.provider_mode == "remote"
    assert called.get("run_name") == "test_remote"


def test_full_pipeline_default_uses_complete_p6_catalog() -> None:
    catalog = default_arm_catalog(
        input_axt_path="fixture.axt",
        model_config="model.yaml",
        source_content_id="same_source_content",
        seed=13,
    )

    assert _full_p6_arm_ids() == list(catalog.keys())


def test_axiom_full_uses_integrated_llama_server(monkeypatch, tmp_path: Path) -> None:
    called: dict[str, object] = {}

    @contextmanager
    def fake_start_llama_server(config: LlamaServerConfig) -> Iterator[object]:
        called["llama_config"] = config
        called["server_alive"] = True
        try:
            yield SimpleNamespace(
                api_key="local-secret",
                base_url="http://127.0.0.1:54321/v1",
                model_name=config.hf_repo,
                config=SimpleNamespace(request_timeout_seconds=180.0),
            )
        finally:
            called["server_alive"] = False

    def fake_build_corpus(config: object, **kw: object) -> object:
        called.update(kw)
        called["corpus_config"] = config
        called["providers"] = getattr(config, "providers", None)
        called["server_alive_during_corpus"] = called["server_alive"]
        # A path that cannot exist: if the real AXT compiler ever runs here, it fails loudly.
        return SimpleNamespace(
            package_path=tmp_path / "missing" / "dataset.axp",
            manifest=SimpleNamespace(source_count=3, capsule_count=11),
        )

    def fake_run_train(*, dataset: Path, **kw: object) -> Path:
        called["server_alive_during_train"] = called["server_alive"]
        called["train_dataset"] = dataset
        return tmp_path / "run"

    def fake_axt_compile(config: object, *, force: bool = False) -> object:
        called["axt_input_path"] = getattr(config, "input_path", None)
        called["axt_force"] = force
        return SimpleNamespace(manifest=SimpleNamespace(record_count=7))

    # pipeline_cli imports these inside its functions, so patch the source modules.
    monkeypatch.setattr(
        "hcaps.providers.llama_server_runtime.start_llama_server", fake_start_llama_server
    )
    monkeypatch.setattr("hcaps.corpus.builder.build_corpus", fake_build_corpus)
    monkeypatch.setattr("hcaps.axt.compiler.compile_axt", fake_axt_compile)
    monkeypatch.setattr("hcaps.pipeline_cli.run_train", fake_run_train)
    monkeypatch.setattr("hcaps.pipeline_cli.PROJECT_ROOT", tmp_path)

    runner = CliRunner()
    # full: no harvest, default source folder, extraction model, no advanced corpus
    inp = "\n\nggml-org/test-GGUF:Q4_K_M\n\n"
    res = runner.invoke(app, ["full"], input=inp)

    assert res.exit_code == 0, res.output
    llama_config = called["llama_config"]
    assert isinstance(llama_config, LlamaServerConfig)
    assert llama_config.hf_repo == "ggml-org/test-GGUF:Q4_K_M"
    providers = called["providers"]
    assert isinstance(providers, ProviderIngressConfig)
    assert providers.primary.provider_id == "integrated_llama_server"
    assert providers.primary.provider_mode == "local"
    assert providers.primary.model == "ggml-org/test-GGUF:Q4_K_M"
    assert providers.escalation is None
    assert providers.cascade.enabled is False
    assert called["server_alive_during_corpus"] is True
    assert called["server_alive_during_train"] is False
    assert called["axt_force"] is True
    train_dataset = called["train_dataset"]
    assert isinstance(train_dataset, Path)
    assert train_dataset.name.endswith(".axt")


def test_axiom_train_uses_pioneer_config_and_steps(monkeypatch, tmp_path: Path) -> None:
    """`axiom train` derives its config from the pioneer E config + prompt answers."""
    called: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, config: TrainingConfig) -> None:
            called["config"] = config

        def fit(self) -> SimpleNamespace:
            return SimpleNamespace(arm_dir=tmp_path / "arm", run_dir=tmp_path / "run")

    monkeypatch.setattr("hcaps.training.trainer.AxiomTrainer", FakeTrainer)

    runner = CliRunner()
    inp = "artifacts/axt/pioneer.axt\n20\n\n"  # dataset, steps, no advanced
    res = runner.invoke(app, ["train"], input=inp)

    assert res.exit_code == 0, res.output
    config = called["config"]
    assert isinstance(config, TrainingConfig)
    assert config.max_steps == 20
    assert config.compute_budget_config.max_train_steps == 20
    assert config.geometry_provider_kind == "learned"
    assert config.seed == 13  # pioneer default preserved when advanced is skipped
    assert config.input_axt_path is not None
    assert str(config.input_axt_path).endswith("pioneer.axt")
