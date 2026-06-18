"""Runtime AXT batch representation and collator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AxtBatch:
    """A framework-light batch preserving structured AXT groups."""

    records: list[dict[str, Any]]

    @property
    def size(self) -> int:
        return len(self.records)

    def to_dict(self) -> dict[str, Any]:
        keys = sorted({key for record in self.records for key in record})
        return {
            "size": self.size,
            "records": self.records,
            "groups": {key: [record.get(key) for record in self.records] for key in keys},
        }


class AxtBatchCollator:
    """Collate structured AXT records without flattening them to text only."""

    def __call__(self, records: list[Any]) -> AxtBatch:
        normalized = [
            record.to_dict() if hasattr(record, "to_dict") else dict(record) for record in records
        ]
        return AxtBatch(records=normalized)
