"""Inspection and comparison helpers for AXT bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .reader import AxtBundle


def inspect_axt_bundle(path: str | Path) -> dict[str, Any]:
    bundle = AxtBundle(path)
    return {
        "format": bundle.manifest.format,
        "format_version": bundle.manifest.format_version,
        "record_count": bundle.manifest.record_count,
        "split_name": bundle.manifest.split_name,
        "source_format": bundle.manifest.source_format,
        "tensor_groups": bundle.manifest.tensor_groups,
        "mask_summary": bundle.manifest.mask_summary,
        "negative_sample_summary": bundle.manifest.negative_sample_summary,
        "provider_context_summary": bundle.manifest.provider_context_summary,
        "text_projection_summary": bundle.manifest.text_projection_summary,
        "geometry_slot_summary": bundle.manifest.geometry_slot_summary,
        "warnings": bundle.manifest.warnings,
    }


def inspect_tensor_group(path: str | Path, group_name: str) -> dict[str, Any]:
    bundle = AxtBundle(path)
    group = bundle.read_tensor_group(group_name)
    return {
        "group": group_name,
        "tensors": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in sorted(group.items())
        },
    }


def compare_axt_bundles(left: str | Path, right: str | Path) -> dict[str, Any]:
    left_bundle = AxtBundle(left)
    right_bundle = AxtBundle(right)
    left_hashes = left_bundle.manifest.hashes
    right_hashes = right_bundle.manifest.hashes
    all_paths = sorted(set(left_hashes) | set(right_hashes))
    differences = [
        {
            "path": relative_path,
            "left": left_hashes.get(relative_path),
            "right": right_hashes.get(relative_path),
        }
        for relative_path in all_paths
        if left_hashes.get(relative_path) != right_hashes.get(relative_path)
    ]
    return {
        "same": not differences,
        "left_record_count": left_bundle.manifest.record_count,
        "right_record_count": right_bundle.manifest.record_count,
        "differences": differences,
    }
