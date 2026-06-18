"""AXT bundle validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .reader import AxtBundle
from .schema import REQUIRED_TENSOR_GROUPS


def validate_axt_bundle(path: str | Path) -> dict[str, Any]:
    """Validate manifest, hashes, and required tensor groups."""

    issues: list[dict[str, object]] = []
    try:
        bundle = AxtBundle(path)
    except Exception as exc:
        return {"ok": False, "issues": [{"code": "bundle_open_failed", "message": str(exc)}]}

    missing_groups = sorted(set(REQUIRED_TENSOR_GROUPS) - set(bundle.tensor_group_names()))
    for group in missing_groups:
        issues.append({"code": "missing_tensor_group", "group": group})
    for group in bundle.tensor_group_names():
        tensor_path = bundle.tensor_path(group)
        if not tensor_path.exists():
            issues.append({"code": "missing_tensor_file", "group": group, "path": str(tensor_path)})
    hash_report = bundle.validate_hashes()
    hash_issues = hash_report.get("issues", [])
    if isinstance(hash_issues, list):
        issues.extend(hash_issues)
    return {
        "ok": not issues,
        "format": bundle.manifest.format,
        "format_version": bundle.manifest.format_version,
        "record_count": bundle.manifest.record_count,
        "tensor_groups": bundle.manifest.tensor_groups,
        "issues": issues,
    }
