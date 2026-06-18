"""Dataset for Axiom AXC/AXP capsules.

The `AxcTrainingDataset` class loads a stream of capsules from a `.axc`
file or from a canonical capsules file inside an `.axp` package. Each
element of the dataset yields tokenized inputs, attention masks,
labels for next-token prediction, capsule identifiers, claim family
identifiers, and optional side-channel features.

This implementation reads all capsules into memory at construction
time for simplicity. Future versions may stream from disk or use
memory-mapped files. Side channels are numeric features extracted via
`extract_side_channels`. No truth labels are stored in the text or
side channels.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .examples import (
    render_capsule_text,
    render_flat_text,
    render_structured_text,
)
from .side_channels import extract_side_channels
from .tokenizer import TokenizerProtocol


def _iter_capsules_from_axc(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from a `.axc` file.

    The `.axc` format is newline-delimited JSON. Blank lines are
    skipped. Errors in parsing raise ValueError.
    """
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}: {e}") from e


def _find_capsules_file_in_axp(axp_path: Path) -> Path:
    """Locate the canonical capsules file inside an `.axp` package.

    By convention, the capsules file is located at
    `<axp>/data/capsules.axc`. If the file is missing, a
    FileNotFoundError is raised.
    """
    data_dir = axp_path / "data"
    candidate = data_dir / "capsules.axc"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find capsules file at {candidate}. Ensure your AXP package "
            "contains a data/capsules.axc file."
        )
    return candidate


class AxcTrainingDataset:
    """Iterate over AXC capsules and tokenize them for pretraining."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: TokenizerProtocol,
        mode: str = "flat_text",
        max_length: int | None = None,
        include_side_channels: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.include_side_channels = include_side_channels
        p = Path(path)
        if p.suffix == ".axc":
            capsule_iter = _iter_capsules_from_axc(p)
        elif p.suffix == ".axp":
            capsules_path = _find_capsules_file_in_axp(p)
            capsule_iter = _iter_capsules_from_axc(capsules_path)
        else:
            raise ValueError(f"Unsupported file extension for dataset: {p}")
        self._records: list[dict[str, Any]] = list(capsule_iter)

        # Pre-build texts for each record based on the selected mode
        self._texts: list[str] = []
        for cap in self._records:
            if mode == "flat_text":
                txt = render_flat_text(cap)
            elif mode == "structured_text":
                txt = render_structured_text(cap)
            elif mode == "capsule":
                txt = render_capsule_text(cap)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            self._texts.append(txt)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cap = self._records[idx]
        text = self._texts[idx]
        input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        # For next-token prediction, labels are the same as input_ids
        labels = input_ids.copy()
        attention_mask = [1] * len(input_ids)
        item: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "capsule_id": cap.get("capsule_id") or cap.get("ids", {}).get("capsule_id"),
            "claim_family_id": (
                cap.get("claim_family_id")
                or cap.get("ids", {}).get("claim_family_id")
                or cap.get("claim", {}).get("claim_id")
            ),
        }
        if self.include_side_channels:
            item["side_channels"] = extract_side_channels(cap)
        return item
