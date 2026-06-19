from __future__ import annotations

import pytest

from hcaps.providers import llama_server_runtime
from hcaps.providers.llama_server_runtime import (
    HuggingFaceGgufValidationError,
    _hf_cache_status,
    hf_model_id,
    validate_hf_repo_has_gguf,
)


def test_hf_model_id_strips_llama_cpp_quant_suffix() -> None:
    assert hf_model_id("org/model-GGUF:Q4_K_M") == "org/model-GGUF"


def test_validate_hf_repo_has_gguf_accepts_repo_with_gguf(monkeypatch) -> None:
    def fake_payload(model_id: str, *, timeout_seconds: float) -> dict[str, object]:
        return {"siblings": [{"rfilename": "model.Q4_K_M.gguf"}]}

    monkeypatch.setattr(llama_server_runtime, "_fetch_hf_model_payload", fake_payload)

    validate_hf_repo_has_gguf("org/model-GGUF:Q4_K_M")


def test_validate_hf_repo_has_gguf_rejects_safetensors_repo_with_suggestions(monkeypatch) -> None:
    def fake_payload(model_id: str, *, timeout_seconds: float) -> dict[str, object]:
        return {"siblings": [{"rfilename": "model.safetensors"}]}

    def fake_search(query: str, *, timeout_seconds: float) -> list[dict[str, object]]:
        return [
            {"modelId": "unsloth/diffusiongemma-26B-A4B-it-GGUF"},
            {"modelId": "google/diffusiongemma-26B-A4B-it"},
        ]

    monkeypatch.setattr(llama_server_runtime, "_fetch_hf_model_payload", fake_payload)
    monkeypatch.setattr(llama_server_runtime, "_fetch_hf_search_payload", fake_search)

    with pytest.raises(HuggingFaceGgufValidationError) as exc_info:
        validate_hf_repo_has_gguf("google/diffusiongemma-26B-A4B-it")

    message = str(exc_info.value)
    assert "has no .gguf files" in message
    assert "unsloth/diffusiongemma-26B-A4B-it-GGUF" in message


def test_hf_cache_status_reports_in_progress_download(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    blob_dir = tmp_path / "hf" / "hub" / "models--org--model-GGUF" / "blobs"
    blob_dir.mkdir(parents=True)
    (blob_dir / "abc.downloadInProgress").write_bytes(b"x" * 2048)

    status = _hf_cache_status("org/model-GGUF")

    assert "download in progress" in status
    assert "2.0 KiB" in status
