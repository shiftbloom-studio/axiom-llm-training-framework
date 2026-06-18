"""Stable identifier contracts for HoloCapsule records."""

from __future__ import annotations

import re
from typing import Annotated, Final

from pydantic import Field

ID_BODY_PATTERN: Final[str] = r"[A-Za-z0-9][A-Za-z0-9._:-]*"


def prefixed_pattern(prefix: str) -> str:
    """Return a full-match regex for a stable identifier prefix."""

    return rf"^{re.escape(prefix)}{ID_BODY_PATTERN}$"


type CapsuleId = Annotated[str, Field(pattern=prefixed_pattern("cap_"))]
type ChunkId = Annotated[str, Field(pattern=prefixed_pattern("chunk_"))]
type ClaimId = Annotated[str, Field(pattern=prefixed_pattern("claim_"))]
type CommunityId = Annotated[str, Field(pattern=prefixed_pattern("comm_"))]
type ContextId = Annotated[str, Field(pattern=prefixed_pattern("ctx_"))]
type DocumentId = Annotated[str, Field(pattern=prefixed_pattern("doc_"))]
type FamilyId = Annotated[str, Field(pattern=prefixed_pattern("family_"))]
type FileId = Annotated[str, Field(pattern=prefixed_pattern("file_"))]
type ManifestId = Annotated[str, Field(pattern=prefixed_pattern("manifest_"))]
type RelationId = Annotated[str, Field(pattern=prefixed_pattern("rel_"))]
type SourceId = Annotated[str, Field(pattern=prefixed_pattern("src_"))]
type SpanId = Annotated[str, Field(pattern=prefixed_pattern("span_"))]

SCHEMA_VERSION_PATTERN: Final[str] = r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9._-]+)?$"
SHA256_PATTERN: Final[str] = r"^[a-f0-9]{64}$"
