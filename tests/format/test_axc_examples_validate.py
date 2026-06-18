from __future__ import annotations

from pathlib import Path

import pytest

from hcaps.format.streams import read_axc_stream, validate_axc_stream


@pytest.mark.parametrize(
    "filename,expected_count",
    [
        ("minimal_capsules.axc", 1),
        ("full_capsules.axc", 2),
        ("minimal_dataset.axp/data/capsules.axc", 2),
    ],
)
def test_valid_axc_examples_validate(
    axf_examples_dir: Path,
    filename: str,
    expected_count: int,
) -> None:
    path = axf_examples_dir / filename

    report = validate_axc_stream(path)
    capsules = list(read_axc_stream(path))

    assert report.ok
    assert report.valid_count == expected_count
    assert len(capsules) == expected_count
    assert all(capsule.provenance.sources for capsule in capsules)


def test_full_axc_example_contains_expected_relations(axf_examples_dir: Path) -> None:
    capsules = list(read_axc_stream(axf_examples_dir / "full_capsules.axc"))
    relation_types = {
        relation.relation_type for capsule in capsules for relation in capsule.relations
    }

    assert "supports" in relation_types
    assert "supersedes" in relation_types


def test_canonical_axf_examples_use_native_extensions(axf_examples_dir: Path) -> None:
    paths = [path.relative_to(axf_examples_dir).as_posix() for path in axf_examples_dir.rglob("*")]

    assert "minimal_capsules.axc" in paths
    assert "full_capsules.axc" in paths
    assert "minimal_dataset.axp/data/capsules.axc" in paths
    assert "minimal_dataset.axp/data/sources.axsrc" in paths
    assert "minimal_dataset.axp/data/relations.axr" in paths
    assert "minimal_tensor_bundle.axt" in paths
    assert not any(path.endswith(".axc.jsonl") for path in paths)
