from __future__ import annotations

from tests.spec._helpers import SPEC, assert_contains_all, read_spec

FORBIDDEN_TERMS = [
    "truth",
    "is_true",
    "correct",
    "is_correct",
    "factuality",
    "ground_truth",
    "label_truth",
    "proven_true",
]


def test_axf_v1_declares_truth_style_fields_forbidden() -> None:
    text = read_spec("AXF_V1.md")

    assert_contains_all(text, FORBIDDEN_TERMS)
    assert_contains_all(
        text,
        ["evaluation_reference", "gold_reference", "human_verified_reference"],
    )


def test_v1_specs_do_not_register_forbidden_terms_as_fields() -> None:
    for path in SPEC.glob("*_V1*.md"):
        text = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_TERMS:
            assert f"| {term} |" not in text, f"{path.name} registers forbidden field {term}"
            assert f"`{term}` |" not in text, f"{path.name} registers forbidden field {term}"


def test_geometry_slots_are_gauge_invariant_only() -> None:
    text = read_spec("AXT_V1_TENSOR_BUNDLE.md") + read_spec("AXC_OUT_V1_SCHEMA.md")

    assert_contains_all(
        text,
        [
            "curvature_score",
            "holonomy_norm",
            "trace_summary",
            "spectrum_summary",
            "context_lability",
            "Raw connection matrices are not",
        ],
    )
