"""Structured audits for falsification-ready AXC artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

FORBIDDEN_TRUTH_LABELS = {
    "truth",
    "is_true",
    "correct",
    "is_correct",
    "factuality",
    "ground_truth",
    "label_truth",
    "proven_true",
}

FORBIDDEN_RENDERED_TARGET_MARKERS = {
    "future_summary",
    "future_label",
    "target_timestamp",
    "target_only",
    "ground_truth",
    "answer_key",
}


@dataclass(frozen=True)
class AuditFinding:
    """One structured audit finding."""

    audit: str
    severity: str
    message: str
    capsule_id: str | None = None
    path: str | None = None
    value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit": self.audit,
            "severity": self.severity,
            "message": self.message,
            "capsule_id": self.capsule_id,
            "path": self.path,
            "value": self.value,
        }


def audit_temporal_leakage(
    capsules: Iterable[dict[str, Any]], *, strict: bool = False
) -> list[AuditFinding]:
    """Flag predictor-visible source dates later than valid-as-of cutoffs."""

    findings: list[AuditFinding] = []
    for index, capsule in enumerate(capsules):
        capsule_id = _capsule_id(capsule, index)
        cutoff = _valid_as_of(capsule)
        if cutoff is None:
            findings.append(
                AuditFinding(
                    audit="temporal_leakage",
                    severity="error",
                    capsule_id=capsule_id,
                    path="context.temporal_cutoff.cutoff_at",
                    message="Capsule is missing a valid-as-of cutoff.",
                )
            )
            continue

        for path, raw_value in _source_timestamp_values(capsule):
            timestamp = _parse_datetime(raw_value)
            if timestamp is None:
                findings.append(
                    AuditFinding(
                        audit="temporal_leakage",
                        severity="error",
                        capsule_id=capsule_id,
                        path=path,
                        value=str(raw_value),
                        message="Source timestamp could not be parsed.",
                    )
                )
                continue
            if timestamp > cutoff:
                findings.append(
                    AuditFinding(
                        audit="temporal_leakage",
                        severity="error",
                        capsule_id=capsule_id,
                        path=path,
                        value=str(raw_value),
                        message="Predictor-visible source timestamp is after valid_as_of.",
                    )
                )

    _raise_if_strict(findings, strict)
    return findings


def audit_forbidden_truth_labels(
    capsules: Iterable[dict[str, Any]], *, strict: bool = False
) -> list[AuditFinding]:
    """Recursively flag fields whose names encode binary truth labels."""

    findings: list[AuditFinding] = []
    for index, capsule in enumerate(capsules):
        capsule_id = _capsule_id(capsule, index)
        for path, key in _iter_dict_keys(capsule):
            if key.casefold() in FORBIDDEN_TRUTH_LABELS:
                findings.append(
                    AuditFinding(
                        audit="forbidden_truth_labels",
                        severity="error",
                        capsule_id=capsule_id,
                        path=path,
                        value=key,
                        message="Forbidden truth-label field is present.",
                    )
                )

    _raise_if_strict(findings, strict)
    return findings


def audit_future_target_exposure(
    rendered_texts: Iterable[str | dict[str, Any]], *, strict: bool = False
) -> list[AuditFinding]:
    """Flag rendered predictor text that exposes future-only target markers."""

    findings: list[AuditFinding] = []
    for index, rendered in enumerate(rendered_texts):
        if isinstance(rendered, dict):
            text = str(rendered.get("rendered_text", ""))
            capsule_id = rendered.get("capsule_id")
        else:
            text = str(rendered)
            capsule_id = None
        text_lower = text.casefold()
        for marker in sorted(FORBIDDEN_RENDERED_TARGET_MARKERS):
            if marker.casefold() in text_lower:
                findings.append(
                    AuditFinding(
                        audit="future_target_exposure",
                        severity="error",
                        capsule_id=str(capsule_id) if capsule_id else None,
                        path=f"rendered_text[{index}]",
                        value=marker,
                        message="Rendered predictor text exposes a future target marker.",
                    )
                )

    _raise_if_strict(findings, strict)
    return findings


def audit_required_fields(
    capsules: Iterable[dict[str, Any]], *, strict: bool = False
) -> list[AuditFinding]:
    """Flag missing core AXC fields."""

    required = ("claim", "surface_forms", "epistemic_state", "provenance", "context")
    findings: list[AuditFinding] = []
    for index, capsule in enumerate(capsules):
        capsule_id = _capsule_id(capsule, index)
        ids = capsule.get("ids", {})
        if "capsule_id" not in capsule and not (
            isinstance(ids, dict) and ids.get("capsule_id")
        ):
            findings.append(
                AuditFinding(
                    audit="required_fields",
                    severity="error",
                    capsule_id=capsule_id,
                    path="capsule_id",
                    message="Capsule is missing required capsule ID.",
                )
            )
        for field in required:
            if field not in capsule:
                findings.append(
                    AuditFinding(
                        audit="required_fields",
                        severity="error",
                        capsule_id=capsule_id,
                        path=field,
                        message=f"Capsule is missing required field '{field}'.",
                    )
                )

    _raise_if_strict(findings, strict)
    return findings


def audit_provenance_integrity(
    capsules: Iterable[dict[str, Any]], *, strict: bool = False
) -> list[AuditFinding]:
    """Audit provenance presence and minimal source identity."""

    findings: list[AuditFinding] = []
    for index, capsule in enumerate(capsules):
        capsule_id = _capsule_id(capsule, index)
        provenance = _provenance_list(capsule)
        if not provenance:
            findings.append(
                AuditFinding(
                    audit="provenance_integrity",
                    severity="warning",
                    capsule_id=capsule_id,
                    path="provenance",
                    message="Capsule has no predictor-visible provenance records.",
                )
            )
            continue
        for source_index, source in enumerate(provenance):
            expected_groups = {
                "source_id": ("source_id",),
                "source_title": ("source_title", "title"),
                "source_timestamp": ("source_timestamp", "source_date", "published_at"),
            }
            for label, keys in expected_groups.items():
                has_key = any(key in source for key in keys)
                if not has_key and "source_count_proxy_index" not in source:
                    findings.append(
                        AuditFinding(
                            audit="provenance_integrity",
                            severity="warning",
                            capsule_id=capsule_id,
                            path=f"provenance[{source_index}].{label}",
                            message=f"Provenance record is missing '{label}'.",
                        )
                    )

    _raise_if_strict(findings, strict)
    return findings


def audit_relation_integrity(
    capsules: Iterable[dict[str, Any]], *, strict: bool = False
) -> list[AuditFinding]:
    """Audit relation shape without requiring any relation to exist."""

    findings: list[AuditFinding] = []
    for index, capsule in enumerate(capsules):
        capsule_id = _capsule_id(capsule, index)
        relations = capsule.get("relations", [])
        if not isinstance(relations, list):
            findings.append(
                AuditFinding(
                    audit="relation_integrity",
                    severity="error",
                    capsule_id=capsule_id,
                    path="relations",
                    message="Relations field must be a list when present.",
                )
            )
            continue
        for relation_index, relation in enumerate(relations):
            if not isinstance(relation, dict):
                findings.append(
                    AuditFinding(
                        audit="relation_integrity",
                        severity="error",
                        capsule_id=capsule_id,
                        path=f"relations[{relation_index}]",
                        message="Relation record must be an object.",
                    )
                )
                continue
            for key in ("relation_type", "target_claim_id"):
                legacy_ok = key == "relation_type" and "type" in relation
                if key not in relation and not legacy_ok:
                    findings.append(
                        AuditFinding(
                            audit="relation_integrity",
                            severity="warning",
                            capsule_id=capsule_id,
                            path=f"relations[{relation_index}].{key}",
                            message=f"Relation record is missing '{key}'.",
                        )
                    )

    _raise_if_strict(findings, strict)
    return findings


def audit_context_integrity(
    capsules: Iterable[dict[str, Any]], *, strict: bool = False
) -> list[AuditFinding]:
    """Audit context presence and temporal cutoff availability."""

    findings: list[AuditFinding] = []
    for index, capsule in enumerate(capsules):
        capsule_id = _capsule_id(capsule, index)
        context = capsule.get("context")
        if not isinstance(context, dict):
            findings.append(
                AuditFinding(
                    audit="context_integrity",
                    severity="warning",
                    capsule_id=capsule_id,
                    path="context",
                    message="Capsule has no context object.",
                )
            )
            continue
        if _valid_as_of(capsule) is None:
            findings.append(
                AuditFinding(
                    audit="context_integrity",
                    severity="error",
                    capsule_id=capsule_id,
                    path="context.temporal_cutoff.cutoff_at",
                    message="Context is missing a temporal cutoff.",
                )
            )

    _raise_if_strict(findings, strict)
    return findings


def audit_split_integrity(
    split_manifest: dict[str, Any], capsules: Iterable[dict[str, Any]], *, strict: bool = False
) -> list[AuditFinding]:
    """Audit split IDs for duplicates and unknown capsule references."""

    capsule_ids = {_capsule_id(capsule, index) for index, capsule in enumerate(capsules)}
    findings: list[AuditFinding] = []
    for split_name, split_ids in split_manifest.items():
        if not isinstance(split_ids, list):
            findings.append(
                AuditFinding(
                    audit="split_integrity",
                    severity="error",
                    path=split_name,
                    message="Split manifest value must be a list of capsule IDs.",
                )
            )
            continue
        seen: set[str] = set()
        for split_id in split_ids:
            if split_id in seen:
                findings.append(
                    AuditFinding(
                        audit="split_integrity",
                        severity="error",
                        capsule_id=str(split_id),
                        path=split_name,
                        message="Duplicate capsule ID appears in split.",
                    )
                )
            seen.add(str(split_id))
            if str(split_id) not in capsule_ids:
                findings.append(
                    AuditFinding(
                        audit="split_integrity",
                        severity="error",
                        capsule_id=str(split_id),
                        path=split_name,
                        message="Split references a capsule ID not present in input.",
                    )
                )

    _raise_if_strict(findings, strict)
    return findings


def audit_input_capsules(
    capsules: list[dict[str, Any]], *, strict: bool = False
) -> list[AuditFinding]:
    """Run the standard input audit suite."""

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
    _raise_if_strict(findings, strict)
    return findings


def summarize_findings(findings: Iterable[AuditFinding]) -> dict[str, int]:
    """Summarize findings by severity."""

    summary = {"error": 0, "warning": 0, "info": 0}
    for finding in findings:
        summary[finding.severity] = summary.get(finding.severity, 0) + 1
    return summary


def findings_to_dicts(findings: Iterable[AuditFinding]) -> list[dict[str, Any]]:
    """Serialize findings for JSON output."""

    return [finding.to_dict() for finding in findings]


def _capsule_id(capsule: dict[str, Any], index: int) -> str:
    ids = capsule.get("ids", {})
    if capsule.get("capsule_id"):
        return str(capsule["capsule_id"])
    if isinstance(ids, dict) and ids.get("capsule_id"):
        return str(ids["capsule_id"])
    return f"record_{index}"


def _valid_as_of(capsule: dict[str, Any]) -> datetime | None:
    temporal = capsule.get("temporal", {})
    if isinstance(temporal, dict) and temporal.get("valid_as_of"):
        return _parse_datetime(temporal["valid_as_of"])
    context = capsule.get("context", {})
    if isinstance(context, dict):
        cutoff = context.get("temporal_cutoff", {})
        if isinstance(cutoff, dict) and cutoff.get("cutoff_at"):
            return _parse_datetime(cutoff["cutoff_at"])
    return None


def _source_timestamp_values(capsule: dict[str, Any]) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    for index, source in enumerate(_provenance_list(capsule)):
        if source.get("target_only"):
            continue
        for key in ("source_timestamp", "source_date", "published_at", "timestamp"):
            if key in source:
                values.append((f"provenance[{index}].{key}", source[key]))

    surface_forms = capsule.get("surface_forms", {})
    if isinstance(surface_forms, dict):
        spans = surface_forms.get("original_spans") or surface_forms.get("source_spans", [])
        if isinstance(spans, list):
            for index, span in enumerate(spans):
                timestamp = None
                timestamp_key = None
                if isinstance(span, dict):
                    for key in ("source_timestamp", "source_date", "published_at", "timestamp"):
                        if span.get(key):
                            timestamp = span[key]
                            timestamp_key = key
                            break
                if (
                    isinstance(span, dict)
                    and not span.get("target_only")
                    and timestamp is not None
                    and timestamp_key is not None
                ):
                    values.append(
                        (
                            f"surface_forms.source_spans[{index}].{timestamp_key}",
                            timestamp,
                        )
                    )
    return values


def _provenance_list(capsule: dict[str, Any]) -> list[dict[str, Any]]:
    provenance = capsule.get("provenance", [])
    if isinstance(provenance, list):
        return [record for record in provenance if isinstance(record, dict)]
    if isinstance(provenance, dict):
        sources = provenance.get("sources", [])
        if isinstance(sources, list):
            return [record for record in sources if isinstance(record, dict)]
    return []


def _iter_dict_keys(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield child_path, str(key)
            yield from _iter_dict_keys(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_dict_keys(child, f"{path}[{index}]")


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _raise_if_strict(findings: list[AuditFinding], strict: bool) -> None:
    if strict and any(finding.severity == "error" for finding in findings):
        first = next(finding for finding in findings if finding.severity == "error")
        raise ValueError(f"{first.audit}: {first.message}")
