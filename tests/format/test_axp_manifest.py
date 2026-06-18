from __future__ import annotations

from hcaps.format.identifiers import package_id
from hcaps.format.manifest import AxpConstruction, AxpManifest
from hcaps.utils.time import utc_now


def test_axp_manifest_schema_validates() -> None:
    manifest = AxpManifest(
        package_id=package_id("fixture", "0.1.0"),
        dataset_name="Fixture",
        created_at=utc_now(),
        license="fixture",
        construction=AxpConstruction(builder="test", builder_version="0.1.0"),
    )

    assert manifest.format == "AXP"
    assert manifest.schema_versions.axc == "0.1.0"
