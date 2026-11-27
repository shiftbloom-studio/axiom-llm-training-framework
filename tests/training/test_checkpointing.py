from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from hcaps.training.checkpointing import load_training_checkpoint
from hcaps.training.seeds import capture_rng_state


class _Opaque:
    pass


def test_load_training_checkpoint_accepts_rng_state_by_default(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    torch.save(
        {"model_state": OrderedDict(w=torch.zeros(2)), "rng_state": capture_rng_state()}, path
    )

    payload = load_training_checkpoint(path)

    assert "numpy" in payload["rng_state"]


def test_load_training_checkpoint_rejects_arbitrary_objects_unless_trusted(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    torch.save({"extra": _Opaque()}, path)

    with pytest.raises(Exception, match="Weights only load failed"):
        load_training_checkpoint(path)

    assert isinstance(load_training_checkpoint(path, trusted=True)["extra"], _Opaque)
