"""Batch collation for training.

The TrainingCollator pads sequences in a batch to the same length
using the tokenizer's pad token ID. It also handles padding labels
with ``-100`` so that padding tokens are ignored by loss functions in
many deep learning frameworks. Side channels are stacked into a list
of lists (2D structure) if present.
"""

from __future__ import annotations

from typing import Any


class TrainingCollator:
    """Callable that collates a list of dataset items into a batch."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Compute max sequence length in the batch
        max_len = max(len(item["input_ids"]) for item in batch)
        padded_input_ids: list[list[int]] = []
        padded_attention: list[list[int]] = []
        padded_labels: list[list[int]] = []
        capsule_ids: list[str | None] = []
        claim_family_ids: list[str | None] = []
        side_channels: list[list[float]] | None = [] if "side_channels" in batch[0] else None
        for item in batch:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            labels = item["labels"]
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_attention.append(mask + [0] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)
            capsule_ids.append(item.get("capsule_id"))
            claim_family_ids.append(item.get("claim_family_id"))
            if side_channels is not None:
                side_channels.append(item["side_channels"])
        batch_dict: dict[str, Any] = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention,
            "labels": padded_labels,
            "capsule_id": capsule_ids,
            "claim_family_id": claim_family_ids,
        }
        if side_channels is not None:
            batch_dict["side_channels"] = side_channels
        return batch_dict
