"""Orchestration for torch-free falsification harness runs."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

import hcaps
from hcaps.falsification.arms import (
    ExperimentArmArtifact,
    ExperimentArmName,
    all_arm_names,
    default_arm_config,
    parse_arm_names,
)
from hcaps.falsification.audits import (
    AuditFinding,
    audit_context_integrity,
    audit_forbidden_truth_labels,
    audit_future_target_exposure,
    audit_input_capsules,
    audit_provenance_integrity,
    audit_relation_integrity,
    audit_required_fields,
    audit_temporal_leakage,
    findings_to_dicts,
    summarize_findings,
)
from hcaps.falsification.controls import (
    build_popularity_frequency_control,
    build_relation_ablation,
    remove_context,
    remove_provenance,
    remove_relations,
    shuffle_contexts,
    shuffle_provenance,
)
from hcaps.falsification.manifests import (
    ArmManifest,
    FalsificationRunManifest,
    sha256_file,
    sha256_path,
    stable_run_id,
    utc_now_iso,
    write_json,
)
from hcaps.falsification.metrics import compute_dataset_metrics
from hcaps.falsification.splits import apply_split, load_split_ids, make_temporal_holdout
from hcaps.schema.capsule import HoloCapsule
from hcaps.training_bridge.examples import (
    render_capsule_text,
    render_flat_text,
    render_structured_text,
)
from hcaps.training_bridge.tokenizer import WhitespaceTokenizer


@dataclass(frozen=True)
class FalsificationRunConfig:
    """Configuration for one falsification harness run."""

    input_path: str | Path
    output_dir: str | Path
    arms: list[str | ExperimentArmName] | str = field(
        default_factory=lambda: [
            "flat_text",
            "structured_text",
            "capsule_text",
        ]
    )
    seed: int = 13
    tokenizer_path: str | Path | None = None
    max_length: int | None = None
    include_side_channels: bool = True
    strict_audits: bool = False
    split_name: str | None = None
    allow_all_without_split: bool = False
    temporal_cutoff_date: str | None = None


@dataclass(frozen=True)
class FalsificationRunResult:
    """In-memory result returned by the runner."""

    manifest: FalsificationRunManifest
    manifest_path: Path
    run_dir: Path
    arm_artifacts: list[ExperimentArmArtifact]
    input_audit_findings: list[AuditFinding]
    arm_audit_findings: dict[str, list[AuditFinding]]
    arm_metrics: dict[str, dict[str, int | float]]


def run_falsification_harness(config: FalsificationRunConfig) -> FalsificationRunResult:
    """Generate deterministic falsification arms, audits, metrics, and manifests."""

    input_path = Path(config.input_path)
    output_dir = Path(config.output_dir)
    arm_names = _normalize_arm_names(config.arms)
    capsules, source_format, capsules_path = load_capsules(input_path)
    input_hash = sha256_path(input_path)

    if source_format == "axp":
        if config.split_name:
            split_ids = load_split_ids(input_path, config.split_name)
            capsules = apply_split(capsules, split_ids)
        elif not config.allow_all_without_split:
            raise ValueError(
                "AXP input requires split_name unless allow_all_without_split=True is set."
            )

    input_findings = audit_input_capsules(capsules, strict=config.strict_audits)
    run_id = stable_run_id(input_hash, [name.value for name in arm_names], config.seed)
    run_dir = output_dir / "artifacts" / "falsification" / run_id
    (run_dir / "audits").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "arms").mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(config.tokenizer_path)
    warnings: list[str] = []
    arm_manifests: list[ArmManifest] = []
    arm_artifacts: list[ExperimentArmArtifact] = []
    arm_findings: dict[str, list[AuditFinding]] = {}
    arm_metrics: dict[str, dict[str, int | float]] = {}

    for arm_name in arm_names:
        artifact, arm_manifest, findings, metrics = _generate_arm(
            arm_name=arm_name,
            capsules=capsules,
            input_path=capsules_path,
            run_dir=run_dir,
            seed=config.seed,
            tokenizer=tokenizer,
            include_side_channels=config.include_side_channels,
            strict_audits=config.strict_audits,
            temporal_cutoff_date=config.temporal_cutoff_date,
        )
        if arm_manifest.derived_noncanonical:
            warnings.append(f"{arm_name.value} produced a derived_noncanonical artifact.")
        arm_artifacts.append(artifact)
        arm_manifests.append(arm_manifest)
        arm_findings[arm_name.value] = findings
        arm_metrics[arm_name.value] = metrics

    input_audit_payload = {
        "summary": summarize_findings(input_findings),
        "findings": findings_to_dicts(input_findings),
    }
    arm_audits_payload = {
        name: {"summary": summarize_findings(findings), "findings": findings_to_dicts(findings)}
        for name, findings in arm_findings.items()
    }
    write_json(run_dir / "audits" / "input_audit.json", input_audit_payload)
    write_json(run_dir / "audits" / "arm_audits.json", arm_audits_payload)
    write_json(run_dir / "metrics" / "arm_metrics.json", arm_metrics)

    manifest = FalsificationRunManifest(
        run_id=run_id,
        created_at=utc_now_iso(),
        axiom_version=hcaps.__version__,
        input_path=str(input_path),
        input_hash=input_hash,
        source_format=source_format,
        arms=arm_manifests,
        audits={"input": input_audit_payload, "arms": arm_audits_payload},
        metrics=arm_metrics,
        deterministic_seed=config.seed,
        code_version_or_commit=_git_commit(),
        warnings=warnings,
        notes=[
            "Falsification artifacts contain dataset diagnostics only.",
            "No model training or benchmark claims were run.",
        ],
    )
    manifest_path = run_dir / "manifest.json"
    manifest.save(manifest_path)

    return FalsificationRunResult(
        manifest=manifest,
        manifest_path=manifest_path,
        run_dir=run_dir,
        arm_artifacts=arm_artifacts,
        input_audit_findings=input_findings,
        arm_audit_findings=arm_findings,
        arm_metrics=arm_metrics,
    )


def load_capsules(input_path: str | Path) -> tuple[list[dict[str, Any]], str, Path]:
    """Load raw AXC dictionaries from an AXC file or AXP package directory."""

    path = Path(input_path)
    if path.suffix == ".axc" and path.is_file():
        return _read_axc(path), "axc", path
    if path.suffix == ".axp" and path.is_dir():
        capsules_path = path / "data" / "capsules.axc"
        if not capsules_path.exists():
            raise FileNotFoundError(f"AXP package is missing {capsules_path}")
        return _read_axc(capsules_path), "axp", capsules_path
    raise ValueError(f"Unsupported falsification input path: {path}")


def _generate_arm(
    *,
    arm_name: ExperimentArmName,
    capsules: list[dict[str, Any]],
    input_path: Path,
    run_dir: Path,
    seed: int,
    tokenizer: Any | None,
    include_side_channels: bool,
    strict_audits: bool,
    temporal_cutoff_date: str | None,
) -> tuple[ExperimentArmArtifact, ArmManifest, list[AuditFinding], dict[str, int | float]]:
    arm_dir = run_dir / "arms" / arm_name.value
    arm_dir.mkdir(parents=True, exist_ok=True)
    arm_config = default_arm_config(arm_name, input_path=input_path, output_path=arm_dir, seed=seed)
    transformed, capsule_artifact_required, notes = _transform_for_arm(
        arm_name,
        capsules,
        seed=seed,
        temporal_cutoff_date=temporal_cutoff_date,
    )
    rendered_records = _render_records(transformed, arm_name, arm_config.render_mode)
    rendered_path = arm_dir / "rendered.jsonl"
    _write_jsonl(rendered_path, rendered_records)

    capsule_artifact_path: Path | None = None
    capsule_artifact_hash: str | None = None
    derived_noncanonical = False
    if capsule_artifact_required:
        canonical = _all_canonical(transformed)
        capsule_artifact_path = arm_dir / ("capsules.axc" if canonical else "capsules.jsonl")
        _write_jsonl(capsule_artifact_path, transformed)
        capsule_artifact_hash = sha256_file(capsule_artifact_path)
        derived_noncanonical = not canonical

    arm_include_side_channels = include_side_channels and arm_name not in {
        ExperimentArmName.CAPSULE_NO_SIDE_CHANNELS,
        ExperimentArmName.RELATION_ABLATION,
    }
    metrics = compute_dataset_metrics(
        transformed,
        rendered_records,
        tokenizer=tokenizer,
        include_side_channels=arm_include_side_channels,
    )
    findings = _audit_arm(transformed, rendered_records, strict=strict_audits)
    audit_summary = summarize_findings(findings)

    arm_manifest = ArmManifest(
        arm_name=arm_name.value,
        artifact_path=str(rendered_path),
        artifact_hash=sha256_file(rendered_path),
        record_count=len(transformed),
        transform_type=arm_config.control_type,
        render_mode=arm_config.render_mode,
        included_fields=list(arm_config.included_fields),
        excluded_fields=list(arm_config.excluded_fields),
        audit_summary=audit_summary,
        metrics_summary=metrics,
        derived_noncanonical=derived_noncanonical,
        capsule_artifact_path=str(capsule_artifact_path) if capsule_artifact_path else None,
        capsule_artifact_hash=capsule_artifact_hash,
        notes=list(arm_config.notes) + notes,
    )
    arm_manifest_path = arm_dir / "arm_manifest.json"
    write_json(arm_manifest_path, arm_manifest.to_dict())

    artifact = ExperimentArmArtifact(
        name=arm_name,
        output_dir=arm_dir,
        rendered_path=rendered_path,
        capsule_artifact_path=capsule_artifact_path,
        arm_manifest_path=arm_manifest_path,
        record_count=len(transformed),
        derived_noncanonical=derived_noncanonical,
    )
    return artifact, arm_manifest, findings, metrics


def _transform_for_arm(
    arm_name: ExperimentArmName,
    capsules: list[dict[str, Any]],
    *,
    seed: int,
    temporal_cutoff_date: str | None,
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    passthrough_arms = {
        ExperimentArmName.FLAT_TEXT,
        ExperimentArmName.STRUCTURED_TEXT,
        ExperimentArmName.CAPSULE_TEXT,
        ExperimentArmName.CAPSULE_NO_SIDE_CHANNELS,
    }
    if arm_name in passthrough_arms:
        return _copy_capsules(capsules), False, []

    transform_map: dict[
        ExperimentArmName,
        tuple[Callable[[], list[dict[str, Any]]], list[str]],
    ] = {
        ExperimentArmName.CAPSULE_NO_PROVENANCE: (
            lambda: [remove_provenance(capsule) for capsule in capsules],
            [],
        ),
        ExperimentArmName.CAPSULE_NO_RELATIONS: (
            lambda: [remove_relations(capsule) for capsule in capsules],
            [],
        ),
        ExperimentArmName.CAPSULE_NO_CONTEXT: (
            lambda: [remove_context(capsule) for capsule in capsules],
            [],
        ),
        ExperimentArmName.CONTEXT_SHUFFLE: (
            lambda: shuffle_contexts(capsules, seed),
            [f"context_seed={seed}"],
        ),
        ExperimentArmName.PROVENANCE_SHUFFLE: (
            lambda: shuffle_provenance(capsules, seed),
            [f"provenance_seed={seed}"],
        ),
        ExperimentArmName.POPULARITY_FREQUENCY_CONTROL: (
            lambda: build_popularity_frequency_control(capsules),
            ["derived_noncanonical=True"],
        ),
        ExperimentArmName.RELATION_ABLATION: (
            lambda: build_relation_ablation(capsules),
            [],
        ),
    }

    if arm_name == ExperimentArmName.TEMPORAL_HOLDOUT:
        return _build_temporal_holdout_arm(capsules, temporal_cutoff_date)
    if arm_name not in transform_map:
        raise ValueError(f"Unsupported arm: {arm_name}")

    transform, notes = transform_map[arm_name]
    return transform(), True, notes


def _render_records(
    capsules: list[dict[str, Any]],
    arm_name: ExperimentArmName,
    render_mode: str | None,
) -> list[dict[str, Any]]:
    return [
        {
            "capsule_id": _capsule_id(capsule, index),
            "claim_id": _claim_id(capsule),
            "arm": arm_name.value,
            "render_mode": render_mode,
            "rendered_text": _render_text(capsule, arm_name, render_mode),
        }
        for index, capsule in enumerate(capsules)
    ]


def _render_text(
    capsule: dict[str, Any],
    arm_name: ExperimentArmName,
    render_mode: str | None,
) -> str:
    if arm_name == ExperimentArmName.POPULARITY_FREQUENCY_CONTROL:
        base = render_flat_text(capsule)
        control = capsule.get("popularity_frequency_control", {})
        if isinstance(control, dict):
            return "\n".join(
                [
                    base,
                    f"Source count proxy: {control.get('source_count_proxy', 0)}",
                    "Claim-family frequency proxy: "
                    f"{control.get('claim_family_frequency_proxy', 0)}",
                ]
            ).strip()
        return base
    if render_mode == "flat_text":
        return render_flat_text(capsule)
    if render_mode == "structured_text":
        return render_structured_text(capsule)
    return render_capsule_text(capsule)


def _audit_arm(
    capsules: list[dict[str, Any]],
    rendered_records: list[dict[str, Any]],
    *,
    strict: bool,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for audit in (
        audit_required_fields,
        audit_temporal_leakage,
        audit_forbidden_truth_labels,
        audit_provenance_integrity,
        audit_relation_integrity,
        audit_context_integrity,
    ):
        findings.extend(audit(capsules, strict=False))
    findings.extend(audit_future_target_exposure(rendered_records, strict=False))
    if strict and any(finding.severity == "error" for finding in findings):
        first = next(finding for finding in findings if finding.severity == "error")
        raise ValueError(f"{first.audit}: {first.message}")
    return findings


def _normalize_arm_names(values: list[str | ExperimentArmName] | str) -> list[ExperimentArmName]:
    if isinstance(values, str):
        return parse_arm_names(values)
    if not values:
        return all_arm_names()
    normalized = [
        value.value if isinstance(value, ExperimentArmName) else value for value in values
    ]
    return parse_arm_names(normalized)


def _copy_capsules(capsules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [deepcopy(capsule) for capsule in capsules]


def _build_temporal_holdout_arm(
    capsules: list[dict[str, Any]],
    temporal_cutoff_date: str | None,
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    cutoff = temporal_cutoff_date or _default_temporal_cutoff(capsules)
    split = make_temporal_holdout(capsules, cutoff)
    return split["holdout"], False, [f"temporal_holdout_cutoff={cutoff}"]


def _read_axc(path: Path) -> list[dict[str, Any]]:
    capsules: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: AXC records must be JSON objects.")
        capsules.append(payload)
    return capsules


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _all_canonical(capsules: list[dict[str, Any]]) -> bool:
    if not capsules:
        return True
    for capsule in capsules:
        try:
            HoloCapsule.model_validate(capsule)
        except ValidationError, ValueError:
            return False
    return True


def _load_tokenizer(path: str | Path | None) -> WhitespaceTokenizer | None:
    if path is None:
        return None
    return WhitespaceTokenizer.load(path)


def _capsule_id(capsule: dict[str, Any], index: int) -> str:
    ids = capsule.get("ids", {})
    if capsule.get("capsule_id"):
        return str(capsule["capsule_id"])
    if isinstance(ids, dict) and ids.get("capsule_id"):
        return str(ids["capsule_id"])
    return f"record_{index}"


def _claim_id(capsule: dict[str, Any]) -> str | None:
    claim = capsule.get("claim", {})
    if isinstance(claim, dict) and claim.get("claim_id"):
        return str(claim["claim_id"])
    ids = capsule.get("ids", {})
    if isinstance(ids, dict) and ids.get("claim_family_id"):
        return str(ids["claim_family_id"])
    return None


def _default_temporal_cutoff(capsules: list[dict[str, Any]]) -> str:
    cutoffs: list[str] = []
    for capsule in capsules:
        context = capsule.get("context", {})
        if isinstance(context, dict):
            temporal = context.get("temporal_cutoff", {})
            if isinstance(temporal, dict) and temporal.get("cutoff_at"):
                cutoffs.append(str(temporal["cutoff_at"]))
    if not cutoffs:
        raise ValueError("Cannot infer temporal holdout cutoff: no valid_as_of values found.")
    cutoffs.sort()
    return cutoffs[max(0, (len(cutoffs) // 2) - 1)]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return result.stdout.strip() or None
