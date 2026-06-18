"""Mandatory secondary text-projection tensor compilation."""

from __future__ import annotations

from typing import Any

import numpy as np

from hcaps.training_bridge.examples import (
    render_capsule_text,
    render_flat_text,
    render_structured_text,
)
from hcaps.training_bridge.tokenizer import WhitespaceTokenizer

from .schema import SPECIAL_TOKENS, TEXT_RENDER_MODES, TensorGroup, as_i64
from .vocab import VocabularyRegistry


def compile_text_projection(
    capsules: list[Any],
    vocab: VocabularyRegistry,
    *,
    render_modes: list[str],
    max_text_length: int,
) -> tuple[TensorGroup, dict[str, object], dict[str, object]]:
    """Compile flat and structured text projections without making them native substrate."""

    modes = [mode for mode in render_modes if mode in TEXT_RENDER_MODES]
    if not modes:
        modes = list(TEXT_RENDER_MODES)
    raw_records = [capsule.model_dump(mode="json") for capsule in capsules]
    rendered_by_mode = {
        "flat_text": [render_flat_text(record) for record in raw_records],
        "structured_text": [render_structured_text(record) for record in raw_records],
        "capsule_text": [render_capsule_text(record) for record in raw_records],
    }
    tokenizer = WhitespaceTokenizer().fit(
        list(SPECIAL_TOKENS) + [text for mode in modes for text in rendered_by_mode[mode]]
    )
    arrays: dict[str, np.ndarray] = {}
    for mode in TEXT_RENDER_MODES:
        encoded = [_encode(tokenizer, text, max_text_length) for text in rendered_by_mode[mode]]
        arrays[f"{mode}_input_ids"] = np.asarray([item[0] for item in encoded], dtype=np.int64)
        arrays[f"{mode}_attention_mask"] = np.asarray([item[1] for item in encoded], dtype=np.int64)
    arrays["text_input_ids"] = arrays["flat_text_input_ids"]
    arrays["text_attention_mask"] = arrays["flat_text_attention_mask"]
    arrays["text_labels"] = arrays["flat_text_input_ids"].copy()
    arrays["text_projection_targets"] = arrays["flat_text_input_ids"].copy()
    arrays["text_loss_mask"] = arrays["flat_text_attention_mask"].copy()
    arrays["text_render_mode"] = as_i64([vocab.lookup("text_render_mode", mode) for mode in modes])
    arrays["special_token_ids"] = as_i64(
        [tokenizer.vocab.get(token.lower(), tokenizer.unk_token_id) for token in SPECIAL_TOKENS]
    )
    summary = {
        "render_modes": modes,
        "max_text_length": max_text_length,
        "tokenizer": "WhitespaceTokenizer",
        "vocab_size": tokenizer.vocab_size,
        "text_projection_is_secondary": True,
        "future_target_exclusion": (
            "future/evaluation-only fields are not rendered by projection functions"
        ),
    }
    sidecar = {
        "tokenizer_vocab": tokenizer.vocab,
        "rendered_text": {mode: texts for mode, texts in rendered_by_mode.items() if mode in modes},
    }
    return arrays, summary, sidecar


def _encode(
    tokenizer: WhitespaceTokenizer, text: str, max_text_length: int
) -> tuple[list[int], list[int]]:
    ids = tokenizer.encode(text, max_length=max_text_length)
    padded = ids + [tokenizer.pad_token_id] * (max_text_length - len(ids))
    mask = [1] * len(ids) + [0] * (max_text_length - len(ids))
    return padded[:max_text_length], mask[:max_text_length]
