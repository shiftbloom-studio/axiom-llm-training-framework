"""Deterministic document chunking for claim-field substrate builds."""

from __future__ import annotations

import re
from dataclasses import dataclass

from hcaps.ingest.documents import DocumentChunk, SourceDocument
from hcaps.substrate.canonicalize import stable_id
from hcaps.substrate.manifest import SubstrateBuildConfig

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextBlock:
    text: str
    start_char: int
    end_char: int
    section_path: tuple[str, ...]


def chunk_documents(
    documents: list[SourceDocument],
    config: SubstrateBuildConfig,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, config))
    return chunks


def chunk_document(document: SourceDocument, config: SubstrateBuildConfig) -> list[DocumentChunk]:
    """Split a document into stable chunks using headings, paragraphs, and sentences."""

    chunks: list[DocumentChunk] = []
    for block in _semantic_blocks(document.text):
        chunks.extend(_split_block(document, block, config, len(chunks)))
    return chunks


def _semantic_blocks(text: str) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    section_path: list[str] = []
    paragraph_lines: list[str] = []
    paragraph_start: int | None = None
    cursor = 0

    def flush(end: int) -> None:
        nonlocal paragraph_lines, paragraph_start
        if paragraph_start is None:
            return
        paragraph = "\n".join(paragraph_lines).strip()
        if paragraph:
            blocks.append(
                TextBlock(
                    text=paragraph,
                    start_char=paragraph_start,
                    end_char=end,
                    section_path=tuple(section_path),
                )
            )
        paragraph_lines = []
        paragraph_start = None

    for line in text.splitlines(keepends=True):
        line_without_newline = line.rstrip("\n")
        stripped = line_without_newline.strip()
        match = HEADING_PATTERN.match(stripped)
        if match:
            flush(cursor)
            level = len(match.group(1))
            heading = match.group(2).strip()
            section_path = section_path[: level - 1]
            section_path.append(heading)
        elif not stripped:
            flush(cursor)
        else:
            if paragraph_start is None:
                paragraph_start = (
                    cursor + len(line_without_newline) - len(line_without_newline.lstrip())
                )
            paragraph_lines.append(line_without_newline.strip())
        cursor += len(line)
    flush(len(text))

    if not blocks and text.strip():
        start = text.index(text.strip()[0])
        blocks.append(
            TextBlock(text=text.strip(), start_char=start, end_char=len(text), section_path=())
        )
    return blocks


def _split_block(
    document: SourceDocument,
    block: TextBlock,
    config: SubstrateBuildConfig,
    starting_index: int,
) -> list[DocumentChunk]:
    if len(block.text) <= config.max_chunk_chars:
        return [
            _make_chunk(
                document,
                block.text,
                block.start_char,
                block.end_char,
                block.section_path,
                starting_index,
            )
        ]

    chunks: list[DocumentChunk] = []
    sentence_start = 0
    current_text = ""
    current_start = block.start_char
    chunk_index = starting_index

    for raw_sentence in SENTENCE_BOUNDARY_PATTERN.split(block.text):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        local_start = block.text.find(sentence, sentence_start)
        sentence_start = local_start + len(sentence)
        absolute_start = block.start_char + local_start

        if len(sentence) > config.max_chunk_chars:
            if current_text:
                chunks.append(
                    _make_chunk(
                        document,
                        current_text,
                        current_start,
                        current_start + len(current_text),
                        block.section_path,
                        chunk_index,
                    )
                )
                chunk_index += 1
                current_text = ""
            for hard_start in range(0, len(sentence), config.max_chunk_chars):
                part = sentence[hard_start : hard_start + config.max_chunk_chars].strip()
                part_start = absolute_start + hard_start
                chunks.append(
                    _make_chunk(
                        document,
                        part,
                        part_start,
                        part_start + len(part),
                        block.section_path,
                        chunk_index,
                    )
                )
                chunk_index += 1
            continue

        proposed = f"{current_text} {sentence}".strip()
        if current_text and len(proposed) > config.max_chunk_chars:
            chunks.append(
                _make_chunk(
                    document,
                    current_text,
                    current_start,
                    current_start + len(current_text),
                    block.section_path,
                    chunk_index,
                )
            )
            chunk_index += 1
            current_text = sentence
            current_start = absolute_start
        else:
            if not current_text:
                current_start = absolute_start
            current_text = proposed

    if current_text:
        chunks.append(
            _make_chunk(
                document,
                current_text,
                current_start,
                current_start + len(current_text),
                block.section_path,
                chunk_index,
            )
        )

    if config.chunk_overlap_chars <= 0:
        return chunks
    return _apply_overlap(document, chunks, config.chunk_overlap_chars)


def _make_chunk(
    document: SourceDocument,
    text: str,
    start_char: int,
    end_char: int,
    section_path: tuple[str, ...],
    chunk_index: int,
) -> DocumentChunk:
    chunk_id = stable_id("chunk", document.document_id, chunk_index, start_char, end_char, text)
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document.document_id,
        source_id=document.source_id,
        chunk_index=chunk_index,
        text=text,
        start_char=start_char,
        end_char=end_char,
        section_path=list(section_path),
        page_number=document.page_number,
        source_timestamp=document.published_at,
    )


def _apply_overlap(
    document: SourceDocument,
    chunks: list[DocumentChunk],
    overlap_chars: int,
) -> list[DocumentChunk]:
    overlapped: list[DocumentChunk] = []
    for chunk in chunks:
        start = max(0, chunk.start_char - overlap_chars)
        text = document.text[start : chunk.end_char].strip()
        adjusted_start = chunk.end_char - len(text)
        chunk_id = stable_id(
            "chunk", document.document_id, chunk.chunk_index, adjusted_start, chunk.end_char, text
        )
        overlapped.append(
            chunk.model_copy(
                update={
                    "chunk_id": chunk_id,
                    "text": text,
                    "start_char": adjusted_start,
                    "end_char": chunk.end_char,
                }
            )
        )
    return overlapped
