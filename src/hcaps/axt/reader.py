"""Framework-light reader for AXT bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
from safetensors.numpy import load_file

from hcaps.format.hashing import file_hash

from .manifest import load_manifest
from .schema import TensorGroup


class AxtBundle:
    """Lazy reader for one `.axt/` bundle."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            msg = f"AXT bundle does not exist or is not a directory: {self.path}"
            raise FileNotFoundError(msg)
        self.manifest = load_manifest(self.path / "axiom.json")
        self._source_index: dict[str, Any] | None = None

    def tensor_group_names(self) -> list[str]:
        return list(self.manifest.tensor_groups)

    def tensor_path(self, group_name: str) -> Path:
        return self.path / "tensors" / f"{group_name}.safetensors"

    def read_tensor_group(self, group_name: str) -> TensorGroup:
        tensor_path = self.tensor_path(group_name)
        if not tensor_path.exists():
            msg = f"AXT tensor group is missing: {group_name}"
            raise FileNotFoundError(msg)
        return load_file(str(tensor_path))

    def source_index(self) -> dict[str, Any]:
        if self._source_index is None:
            self._source_index = _read_json(self.path / "manifests" / "source_index.json")
        return self._source_index

    def validate_hashes(self) -> dict[str, object]:
        issues: list[dict[str, str]] = []
        for relative_path, expected_hash in self.manifest.hashes.items():
            artifact_path = self.path / relative_path
            if not artifact_path.exists():
                issues.append({"code": "missing_artifact", "path": relative_path})
                continue
            actual = file_hash(artifact_path)
            if actual != expected_hash:
                issues.append(
                    {
                        "code": "hash_mismatch",
                        "path": relative_path,
                        "expected": expected_hash,
                        "actual": actual,
                    }
                )
        return {"ok": not issues, "issues": issues}

    def load_selected_split(self, split_name: str | None = None) -> list[int]:
        split_masks = self.read_tensor_group("split_masks")
        if split_name is None:
            return list(range(self.manifest.record_count))
        key = f"{split_name}_mask"
        if key not in split_masks:
            msg = f"split mask not present in AXT bundle: {split_name}"
            raise KeyError(msg)
        mask = split_masks[key]
        return [index for index, value in enumerate(mask.tolist()) if int(value) == 1]


def read_axt_bundle(path: str | Path) -> AxtBundle:
    return AxtBundle(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict):
        msg = f"expected JSON object: {path}"
        raise ValueError(msg)
    return payload
