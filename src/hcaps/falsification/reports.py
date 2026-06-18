"""Markdown reports for falsification harness manifests."""

from __future__ import annotations

from pathlib import Path

from hcaps.falsification.manifests import FalsificationRunManifest, load_run_manifest


def write_falsification_report(
    manifest: FalsificationRunManifest | str | Path,
    output_path: str | Path,
) -> None:
    """Write a Markdown readiness report for a falsification run."""

    loaded_manifest = load_run_manifest(manifest) if isinstance(manifest, str | Path) else manifest
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_report(loaded_manifest), encoding="utf-8")


def _render_report(manifest: FalsificationRunManifest) -> str:
    lines = [
        "# Axiom Falsification Harness Report",
        "",
        "This report summarizes generated data arms, audits, and dataset diagnostics.",
        "It does not contain model outcomes or training conclusions.",
        "",
        "## Input Summary",
        "",
        f"- Run ID: `{manifest.run_id}`",
        f"- Created at: `{manifest.created_at}`",
        f"- Input path: `{manifest.input_path}`",
        f"- Source format: `{manifest.source_format}`",
        f"- Input hash: `{manifest.input_hash}`",
        f"- Deterministic seed: `{manifest.deterministic_seed}`",
        "",
        "## Arms Generated",
        "",
        "| Arm | Records | Transform | Render Mode | Artifact |",
        "|---|---:|---|---|---|",
    ]
    for arm in manifest.arms:
        lines.append(
            f"| `{arm.arm_name}` | {arm.record_count} | `{arm.transform_type}` | "
            f"`{arm.render_mode or 'none'}` | `{arm.artifact_path}` |"
        )

    lines.extend(
        [
            "",
            "## Audit Findings",
            "",
            "| Scope | Errors | Warnings | Info |",
            "|---|---:|---:|---:|",
        ]
    )
    input_summary = manifest.audits.get("input", {}).get("summary", {})
    lines.append(
        f"| input | {_summary_value(input_summary, 'error')} | "
        f"{_summary_value(input_summary, 'warning')} | {_summary_value(input_summary, 'info')} |"
    )
    arms = manifest.audits.get("arms", {})
    if isinstance(arms, dict):
        for arm_name, payload in arms.items():
            summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
            lines.append(
                f"| `{arm_name}` | {_summary_value(summary, 'error')} | "
                f"{_summary_value(summary, 'warning')} | {_summary_value(summary, 'info')} |"
            )

    lines.extend(["", "## Metric Table", ""])
    metric_keys = _metric_keys(manifest)
    if metric_keys:
        lines.append("| Arm | " + " | ".join(metric_keys) + " |")
        lines.append("|---|" + "|".join("---:" for _ in metric_keys) + "|")
        for arm in manifest.arms:
            values = [str(arm.metrics_summary.get(key, "")) for key in metric_keys]
            lines.append(f"| `{arm.arm_name}` | " + " | ".join(values) + " |")
    else:
        lines.append("No dataset diagnostics were recorded.")

    lines.extend(["", "## Warnings", ""])
    if manifest.warnings:
        lines.extend(f"- {warning}" for warning in manifest.warnings)
    else:
        lines.append("- None recorded.")

    lines.extend(
        [
            "",
            "## Next-Step Interpretation Rules",
            "",
            "- Compare flat text and structured text first; a text-view artifact can "
            "explain later differences.",
            "- Treat ablations as checks that provenance, relations, context, and "
            "side channels are separable signals.",
            "- Treat context and provenance shuffles as tests for accidental label "
            "or source-order shortcuts.",
            "- Treat popularity and frequency controls as confound checks, not epistemic labels.",
            "- Treat any temporal leakage finding as blocking until the source "
            "artifact is repaired.",
            "- Carry these manifests into future P8/P9 planning before choosing "
            "training settings; they are preparation artifacts, not a verdict.",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_value(summary: object, key: str) -> int:
    if isinstance(summary, dict):
        value = summary.get(key, 0)
        return int(value) if isinstance(value, int | float) else 0
    return 0


def _metric_keys(manifest: FalsificationRunManifest) -> list[str]:
    preferred = [
        "record_count",
        "total_token_count",
        "empty_text_count",
        "relation_count",
        "provenance_source_count",
        "unique_claim_family_count",
        "temporal_leakage_count",
        "forbidden_truth_label_count",
        "future_target_exposure_count",
    ]
    present: set[str] = set()
    for arm in manifest.arms:
        present.update(arm.metrics_summary)
    return [key for key in preferred if key in present]
