from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "spec"


def read_spec(name: str) -> str:
    return (SPEC / name).read_text(encoding="utf-8")


def assert_contains_all(text: str, terms: list[str]) -> None:
    normalized_text = " ".join(text.split())
    missing = [
        term for term in terms if term not in text and " ".join(term.split()) not in normalized_text
    ]
    assert not missing, f"missing terms: {missing}"
