"""Tokenizer protocol and baseline whitespace tokenizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Self


class TokenizerProtocol(Protocol):
    """Minimal tokenizer interface consumed by training bridge datasets."""

    pad_token_id: int

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens in the tokenizer vocabulary."""
        ...

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """Encode text into token IDs."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into text."""
        ...


class WhitespaceTokenizer:
    """Simple lowercase tokenizer that splits text on whitespace."""

    pad_token = "<pad>"
    unk_token = "<unk>"
    pad_token_id = 0
    unk_token_id = 1

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self.vocab = vocab or {self.pad_token: self.pad_token_id, self.unk_token: self.unk_token_id}
        self._inverse_vocab = {token_id: token for token, token_id in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def fit(self, texts: list[str]) -> Self:
        """Add tokens from texts to the vocabulary in encounter order."""
        for text in texts:
            for token in self._tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        self._inverse_vocab = {token_id: token for token, token_id in self.vocab.items()}
        return self

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in self._tokenize(text)]
        if max_length is not None:
            return token_ids[:max_length]
        return token_ids

    def decode(self, ids: list[int]) -> str:
        tokens = [
            self._inverse_vocab.get(token_id, self.unk_token)
            for token_id in ids
            if token_id != self.pad_token_id
        ]
        return " ".join(tokens)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump({"vocab": self.vocab}, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        vocab = {str(token): int(token_id) for token, token_id in payload["vocab"].items()}
        return cls(vocab=vocab)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
