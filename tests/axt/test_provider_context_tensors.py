from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle


def test_provider_context_tensors_compile_p1_traces(compiled_p1_corpus: Path) -> None:
    bundle = AxtBundle(compiled_p1_corpus)
    provider = bundle.read_tensor_group("provider_context")

    assert int(provider["provider_context_mask"].sum()) == bundle.manifest.record_count
    assert bundle.manifest.provider_context_summary["trace_records_loaded"] > 0
    assert "provider_config_hash_ref" in provider
    assert "cache_replay_flag" in provider
