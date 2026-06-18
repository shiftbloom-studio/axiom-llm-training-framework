"""AXT bundle writer."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import orjson
from safetensors.numpy import save_file

from hcaps.format.hashing import file_hash

from .config import AxtCompileConfig
from .manifest import AxtArtifact, AxtManifest, write_manifest
from .registry import FieldRegistry
from .schema import TensorGroups
from .vocab import VocabularyRegistry


def write_axt_bundle(
    *,
    output_path: str | Path,
    tensor_groups: TensorGroups,
    config: AxtCompileConfig,
    field_registry: FieldRegistry,
    vocabulary_registry: VocabularyRegistry,
    source_index: dict[str, object],
    input_hash: str,
    source_format: Literal["AXC", "AXP"],
    record_count: int,
    reports: dict[str, object],
    summaries: dict[str, object],
    warnings: list[str],
    force: bool = False,
) -> AxtManifest:
    """Write a complete `.axt/` directory bundle."""

    bundle_path = Path(output_path)
    if bundle_path.exists():
        if not force:
            msg = f"AXT output already exists: {bundle_path} (use force to overwrite)"
            raise FileExistsError(msg)
        if bundle_path.is_dir():
            shutil.rmtree(bundle_path)
        else:
            bundle_path.unlink()
    for directory in ("tensors", "registries", "manifests", "reports"):
        (bundle_path / directory).mkdir(parents=True, exist_ok=True)

    artifacts: list[AxtArtifact] = []
    for group_name, tensors in tensor_groups.items():
        tensor_path = bundle_path / "tensors" / f"{group_name}.safetensors"
        save_file(tensors, str(tensor_path), metadata={"format": "AXT", "group": group_name})
        artifacts.append(
            _artifact(bundle_path, tensor_path, "tensor_group", group_name, sorted(tensors))
        )

    field_registry_path = bundle_path / "registries" / "field_registry.json"
    _write_json(field_registry_path, field_registry.to_json_dict())
    artifacts.append(_artifact(bundle_path, field_registry_path, "field_registry", None, []))

    vocabulary_registry_path = bundle_path / "registries" / "vocabulary_registry.json"
    _write_json(vocabulary_registry_path, vocabulary_registry.to_json_dict())
    artifacts.append(
        _artifact(bundle_path, vocabulary_registry_path, "vocabulary_registry", None, [])
    )

    config_path = bundle_path / "manifests" / "compile_config.json"
    _write_json(config_path, config.model_dump(mode="json"))
    artifacts.append(_artifact(bundle_path, config_path, "compile_config", None, []))

    source_index_path = bundle_path / "manifests" / "source_index.json"
    _write_json(source_index_path, source_index)
    artifacts.append(_artifact(bundle_path, source_index_path, "source_index", None, []))

    tensor_manifest_path = bundle_path / "manifests" / "tensor_manifest.json"
    _write_json(
        tensor_manifest_path,
        {
            "tensor_groups": {
                group_name: sorted(tensors)
                for group_name, tensors in sorted(tensor_groups.items(), key=lambda item: item[0])
            }
        },
    )
    artifacts.append(_artifact(bundle_path, tensor_manifest_path, "tensor_manifest", None, []))

    for report_name, report in reports.items():
        report_path = bundle_path / "reports" / f"{report_name}.json"
        _write_json(report_path, report)
        artifacts.append(_artifact(bundle_path, report_path, report_name, None, []))

    hashes = {artifact.relative_path: artifact.sha256 for artifact in artifacts}
    hashes_path = bundle_path / "manifests" / "hashes.json"
    _write_json(hashes_path, hashes)
    hashes_artifact = _artifact(bundle_path, hashes_path, "hashes", None, [])
    artifacts.append(hashes_artifact)
    hashes[hashes_artifact.relative_path] = hashes_artifact.sha256

    manifest = AxtManifest.create(
        input_path=str(config.input_path),
        input_hash=input_hash,
        source_format=source_format,
        field_registry_hash=field_registry.sha256,
        vocabulary_registry_hash=vocabulary_registry.stable_hash(),
        config_hash=config.stable_hash(),
        record_count=record_count,
        split_name=config.split_name,
        tensor_groups=sorted(tensor_groups),
        artifacts=artifacts,
        hashes=hashes,
        mask_summary=_summary_dict(summaries, "mask_summary"),
        negative_sample_summary=_summary_dict(summaries, "negative_sample_summary"),
        provider_context_summary=_summary_dict(summaries, "provider_context_summary"),
        text_projection_summary=_object_dict(summaries, "text_projection_summary"),
        geometry_slot_summary=_summary_dict(summaries, "geometry_slot_summary"),
        warnings=warnings,
    )
    write_manifest(bundle_path / "axiom.json", manifest)
    return manifest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n"
    )


def _artifact(
    bundle_path: Path,
    path: Path,
    role: str,
    tensor_group: str | None,
    tensors: list[str],
) -> AxtArtifact:
    relative_path = path.relative_to(bundle_path).as_posix()
    return AxtArtifact(
        relative_path=relative_path,
        role=role,
        sha256=file_hash(path),
        byte_size=path.stat().st_size,
        tensor_group=tensor_group,
        tensors=tensors,
    )


def _summary_dict(summaries: dict[str, object], key: str) -> dict[str, int]:
    value = summaries.get(key, {})
    if isinstance(value, dict):
        return {str(item_key): int(item_value) for item_key, item_value in value.items()}
    return {}


def _object_dict(summaries: dict[str, object], key: str) -> dict[str, object]:
    value = summaries.get(key, {})
    return (
        {str(item_key): item_value for item_key, item_value in value.items()}
        if isinstance(value, dict)
        else {}
    )
