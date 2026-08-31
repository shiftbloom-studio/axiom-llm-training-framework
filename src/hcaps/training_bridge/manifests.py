"""Training bridge manifest models.

The manifest captures metadata about a processed dataset for
reproducibility. A manifest can be serialized to and from JSON. This
simple implementation uses a dataclass-like pattern without relying
on Pydantic to minimise dependencies at this stage of the project.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TrainingBridgeManifest:
    """Metadata record for a processed AXC/AXP dataset."""

    input_path: str
    input_hash: str | None
    tokenizer_type: str
    tokenizer_vocab_size: int
    mode: str
    max_length: int | None
    include_side_channels: bool
    record_count: int
    created_at: str
    version: str = "0.1.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: Path) -> TrainingBridgeManifest:
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return cls(**data)

    @staticmethod
    def create(  # noqa: PLR0917
        input_path: str,
        input_hash: str | None,
        tokenizer_type: str,
        tokenizer_vocab_size: int,
        mode: str,
        max_length: int | None,
        include_side_channels: bool,
        record_count: int,
    ) -> TrainingBridgeManifest:
        return TrainingBridgeManifest(
            input_path=input_path,
            input_hash=input_hash,
            tokenizer_type=tokenizer_type,
            tokenizer_vocab_size=tokenizer_vocab_size,
            mode=mode,
            max_length=max_length,
            include_side_channels=include_side_channels,
            record_count=record_count,
            created_at=datetime.utcnow().isoformat() + "Z",
        )
