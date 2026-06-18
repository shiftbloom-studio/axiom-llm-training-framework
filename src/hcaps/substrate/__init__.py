"""Claim-field substrate construction.

Exports are loaded lazily so low-level modules such as canonicalization can be
used by ingestion without importing the full builder orchestration layer.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BuildWarning",
    "ClaimFamily",
    "ClaimFieldSubstrateBuilder",
    "SubstrateBuildConfig",
    "SubstrateBuildManifest",
    "SubstrateBuildResult",
    "build_claim_capsules",
    "build_claim_families",
    "build_substrate",
    "read_build_manifest",
]


def __getattr__(name: str) -> Any:
    if name in {"BuildWarning", "SubstrateBuildConfig", "SubstrateBuildManifest"}:
        from hcaps.substrate import manifest  # noqa: PLC0415

        return getattr(manifest, name)
    if name in {
        "ClaimFamily",
        "ClaimFieldSubstrateBuilder",
        "SubstrateBuildResult",
        "build_claim_capsules",
        "build_claim_families",
        "build_substrate",
        "read_build_manifest",
    }:
        from hcaps.substrate import builder  # noqa: PLC0415

        return getattr(builder, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
