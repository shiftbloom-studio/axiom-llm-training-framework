"""Checkpoint save/load helpers for the P4 model stack."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

import hcaps
from hcaps.model.config import P4_MODEL_SCHEMA_VERSION, AxiomModelConfig
from hcaps.model.factory import AxiomStructuredModel
from hcaps.model.parameter_count import parameter_count


def save_model_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: AxiomModelConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "config": config.model_dump(mode="json"),
        "metadata": {
            **(metadata or {}),
            "model_config": config.model_dump(mode="json"),
            "parameter_count": parameter_count(model),
            "axiom_version": hcaps.__version__,
            "p4_model_schema_version": P4_MODEL_SCHEMA_VERSION,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "compatible_axt_spec_version": config.compatible_axt_spec_version,
            "compatible_axc_out_spec_version": config.compatible_axc_out_spec_version,
        },
    }
    torch.save(payload, Path(path))


def load_model_checkpoint(
    path: str | Path,
    *,
    trusted: bool = False,
) -> tuple[torch.nn.Module, AxiomModelConfig, dict[str, Any]]:
    """Load a model checkpoint.

    By default only tensors and plain containers are unpickled (``weights_only=True``).
    Pass ``trusted=True`` to allow arbitrary pickled objects; only do that for
    checkpoints from a source you trust, since unpickling can run code.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=not trusted)
    config = AxiomModelConfig.model_validate(payload["config"])
    model = AxiomStructuredModel(config)
    model.load_state_dict(payload["state_dict"])
    metadata = payload.get("metadata", {})
    return model, config, metadata if isinstance(metadata, dict) else {}
