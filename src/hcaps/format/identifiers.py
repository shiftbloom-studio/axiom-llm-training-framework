"""Validated AXF identifier helpers."""

from __future__ import annotations

import re
from typing import Annotated, Final

from pydantic import Field

from hcaps.format.hashing import sha256_text

HASH_PATTERN: Final[str] = r"[a-f0-9]{16,64}"
SLUG_PATTERN: Final[str] = r"[a-z0-9][a-z0-9._-]*"
DATE_OR_CONTEXT_PATTERN: Final[str] = r"[0-9]{4}-[0-9]{2}-[0-9]{2}|[a-z0-9._-]+"

AXC_CAPSULE_ID_PATTERN: Final[str] = rf"^axc:sha256:{HASH_PATTERN}$"
CLAIM_FAMILY_ID_PATTERN: Final[str] = rf"^claimfam:{SLUG_PATTERN}:{HASH_PATTERN}$"
CLAIM_STATE_ID_PATTERN: Final[str] = rf"^claimstate:{DATE_OR_CONTEXT_PATTERN}:{HASH_PATTERN}$"
SOURCE_ID_PATTERN: Final[str] = rf"^src:{SLUG_PATTERN}:{HASH_PATTERN}$"
SPAN_ID_PATTERN: Final[str] = rf"^span:{HASH_PATTERN}$"
RELATION_ID_PATTERN: Final[str] = rf"^rel:{HASH_PATTERN}$"
CONTEXT_ID_PATTERN: Final[str] = rf"^ctx:{SLUG_PATTERN}$|^ctx:sha256:{HASH_PATTERN}$"
PACKAGE_ID_PATTERN: Final[str] = rf"^axp:{SLUG_PATTERN}:[A-Za-z0-9._:-]+$"

type AxcCapsuleId = Annotated[str, Field(pattern=AXC_CAPSULE_ID_PATTERN)]
type ClaimFamilyId = Annotated[str, Field(pattern=CLAIM_FAMILY_ID_PATTERN)]
type ClaimStateId = Annotated[str, Field(pattern=CLAIM_STATE_ID_PATTERN)]
type SourceId = Annotated[str, Field(pattern=SOURCE_ID_PATTERN)]
type SpanId = Annotated[str, Field(pattern=SPAN_ID_PATTERN)]
type RelationId = Annotated[str, Field(pattern=RELATION_ID_PATTERN)]
type ContextId = Annotated[str, Field(pattern=CONTEXT_ID_PATTERN)]
type PackageId = Annotated[str, Field(pattern=PACKAGE_ID_PATTERN)]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.casefold()).strip("-._")
    return slug or "unknown"


def axc_id(*parts: object) -> str:
    return f"axc:sha256:{sha256_text(*parts)}"


def claim_family_id(slug: str, *parts: object) -> str:
    return f"claimfam:{slugify(slug)}:{sha256_text(*parts)[:24]}"


def claim_state_id(valid_as_of: str, *parts: object) -> str:
    return f"claimstate:{slugify(valid_as_of)}:{sha256_text(*parts)[:24]}"


def source_id(kind: str, *parts: object) -> str:
    return f"src:{slugify(kind)}:{sha256_text(*parts)[:24]}"


def span_id(*parts: object) -> str:
    return f"span:{sha256_text(*parts)[:24]}"


def relation_id(*parts: object) -> str:
    return f"rel:{sha256_text(*parts)[:24]}"


def context_id(slug: str | None = None, *parts: object) -> str:
    if slug:
        return f"ctx:{slugify(slug)}"
    return f"ctx:sha256:{sha256_text(*parts)[:24]}"


def package_id(slug: str, version_or_hash: str) -> str:
    return f"axp:{slugify(slug)}:{version_or_hash}"
