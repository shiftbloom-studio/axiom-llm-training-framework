"""Source registry helpers for corpus artifacts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hcaps.ingest.documents import SourceDocument


class SourceRegistryEntry(BaseModel):
    """Manifestable source registry entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    raw_hash: str = Field(min_length=64, max_length=64)
    normalized_hash: str = Field(min_length=64, max_length=64)
    license: str = "unknown"
    publication_date: str | None = None
    retrieval_date: str | None = None
    source_path: str = Field(min_length=1)
    source_url: str | None = None
    source_type: str = Field(min_length=1)
    provider_extractor_provenance: list[str] = Field(default_factory=list)
    temporal_cutoff_policy: str = "predictor_side_sources_must_not_exceed_valid_as_of"
    lateral_context: dict[str, object] = Field(default_factory=dict)


def build_source_registry(
    documents: list[SourceDocument],
    *,
    provider_ids: list[str],
) -> list[SourceRegistryEntry]:
    """Build stable source registry records from normalized documents."""

    return [
        SourceRegistryEntry(
            source_id=document.source_id,
            document_id=document.document_id,
            raw_hash=document.raw_sha256,
            normalized_hash=document.normalized_text_sha256,
            license=document.license,
            publication_date=(
                document.published_at.date().isoformat() if document.published_at else None
            ),
            retrieval_date=None,
            source_path=document.source_path,
            source_url=document.source_url,
            source_type=document.media_type,
            provider_extractor_provenance=provider_ids,
            lateral_context={
                "domains": document.domains,
                "reader_name": document.reader_name,
                "provider_identity_recorded": bool(provider_ids),
            },
        )
        for document in sorted(documents, key=lambda item: item.document_id)
    ]


def license_summary(entries: list[SourceRegistryEntry]) -> dict[str, object]:
    """Summarize license values without silently approving restricted material."""

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.license] = counts.get(entry.license, 0) + 1
    restricted = sorted(
        entry.source_id
        for entry in entries
        if entry.license.casefold() in {"restricted", "proprietary"}
    )
    return {
        "licenses": counts,
        "restricted_source_ids": restricted,
        "unknown_license_count": counts.get("unknown", 0),
    }
