from __future__ import annotations

from pathlib import Path

from hcaps.axt.reader import AxtBundle


def test_negative_sample_tensors_compile_p1_negative_pools(compiled_p1_corpus: Path) -> None:
    bundle = AxtBundle(compiled_p1_corpus)
    negatives = bundle.read_tensor_group("negative_samples")

    assert bundle.manifest.negative_sample_summary["negative_records_loaded"] > 0
    assert bundle.manifest.negative_sample_summary["negative_records_compiled"] > 0
    assert int(negatives["negative_loss_eligibility_mask"].sum()) > 0
