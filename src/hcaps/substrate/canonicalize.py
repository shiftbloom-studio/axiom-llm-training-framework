"""Deterministic canonicalization and ID helpers for claim-field builds."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TRAILING_PUNCTUATION = " \t\n\r.,;:!?\"'`()[]{}"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def normalize_text(text: str) -> str:
    """Normalize Unicode, line endings, and whitespace deterministically."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def canonical_fingerprint(text: str) -> str:
    """Return a low-entropy textual fingerprint used for grouping, not identity."""

    normalized = normalize_text(text).casefold().strip(TRAILING_PUNCTUATION)
    tokens = [token for token in TOKEN_PATTERN.findall(normalized) if token not in STOPWORDS]
    return " ".join(tokens)


def token_set(text: str) -> set[str]:
    return set(canonical_fingerprint(text).split())


def token_jaccard(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def edit_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, canonical_fingerprint(left), canonical_fingerprint(right)).ratio()


def claim_similarity(left: str, right: str) -> float:
    """Conservative deterministic claim similarity."""

    return (0.7 * token_jaccard(left, right)) + (0.3 * edit_similarity(left, right))


def stable_hash(parts: Iterable[object], *, length: int = 24) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def stable_id(prefix: str, *parts: object, length: int = 24) -> str:
    return f"{prefix}_{stable_hash(parts, length=length)}"


def safe_slug(value: str) -> str:
    slug = "_".join(TOKEN_PATTERN.findall(value.casefold()))
    return slug[:48] or "unknown"
