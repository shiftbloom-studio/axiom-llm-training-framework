"""Source document, span, and temporal cutoff schema models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from hcaps.schema.identifiers import DocumentId, SourceId, SpanId


class SourceType(StrEnum):
    """Supported source document classes."""

    ARTICLE = "article"
    BOOK = "book"
    DATASET = "dataset"
    PAPER = "paper"
    REVIEW = "review"
    WEBPAGE = "webpage"
    OTHER = "other"


class DocumentSpan(BaseModel):
    """A concrete source-text span used as evidence or a surface form."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    span_id: SpanId
    document_id: DocumentId
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    source_timestamp: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_span_bounds(self) -> DocumentSpan:
        if self.end_char <= self.start_char:
            msg = "DocumentSpan.end_char must be greater than start_char"
            raise ValueError(msg)
        return self


class SourceDocument(BaseModel):
    """A normalized source document record referenced by capsules and manifests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: DocumentId
    source_id: SourceId
    title: str = Field(min_length=1)
    source_type: SourceType
    source_uri: str | None = None
    published_at: AwareDatetime | None = None
    retrieved_at: AwareDatetime | None = None
    license: str = Field(default="unknown", min_length=1)
    content_hash: str | None = None


class TemporalCutoff(BaseModel):
    """Cutoff guard separating model-visible inputs from prediction targets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cutoff_at: AwareDatetime
    input_source_timestamps: list[AwareDatetime] = Field(default_factory=list)
    prediction_target_timestamps: list[AwareDatetime] = Field(default_factory=list)
    future_facing: bool = True

    @model_validator(mode="after")
    def validate_cutoff_direction(self) -> TemporalCutoff:
        for timestamp in self.input_source_timestamps:
            if timestamp > self.cutoff_at:
                msg = (
                    "temporal leakage: input source timestamp "
                    f"{timestamp.isoformat()} is after cutoff {self.cutoff_at.isoformat()}"
                )
                raise ValueError(msg)

        if self.future_facing:
            for timestamp in self.prediction_target_timestamps:
                if timestamp <= self.cutoff_at:
                    msg = (
                        "target leakage: future-facing prediction target timestamp "
                        f"{timestamp.isoformat()} is not after cutoff {self.cutoff_at.isoformat()}"
                    )
                    raise ValueError(msg)
        return self
