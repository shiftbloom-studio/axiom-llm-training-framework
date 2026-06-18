from __future__ import annotations

from pathlib import Path

from hcaps.axt.vocab import compile_vocabulary_registry, load_vocabulary_registry


def test_vocabulary_registry_has_stable_unknown_index() -> None:
    vocab_registry = Path(__file__).resolve().parents[2] / "spec" / "VOCABULARY_REGISTRY_V1.md"
    registry = load_vocabulary_registry(vocab_registry)

    assert registry.lookup("relation_type", "supports") > 1
    assert registry.lookup("relation_type", "not_registered") == 1


def test_vocabulary_registry_appends_dynamic_values() -> None:
    vocab_registry = Path(__file__).resolve().parents[2] / "spec" / "VOCABULARY_REGISTRY_V1.md"
    registry = compile_vocabulary_registry(vocab_registry, dynamic_values={"provider_id": ["det"]})

    assert registry.lookup("provider_id", "det") > 1
    assert registry.stable_hash()
