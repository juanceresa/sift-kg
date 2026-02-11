"""Text chunker — splits documents into overlapping segments for LLM processing.

5000-char chunks with 10% overlap and sentence-boundary-aware splitting.
"""

import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    """A chunk of text with position metadata."""

    text: str
    start_char: int
    end_char: int
    chunk_index: int
    total_chunks: int


# Sentence-ending patterns
_SENTENCE_END = re.compile(r"[.!?]\s+|\n\n|\n(?=[A-Z])")


def chunk_text(
    text: str,
    chunk_size: int = 5000,
    overlap_ratio: float = 0.1,
) -> list[TextChunk]:
    """Split text into overlapping chunks.

    Args:
        text: Full document text
        chunk_size: Target characters per chunk
        overlap_ratio: Fraction of chunk_size to overlap (0.0-0.5)

    Returns:
        List of TextChunk with position metadata
    """
    if not 0.0 <= overlap_ratio <= 0.5:
        raise ValueError("overlap_ratio must be between 0.0 and 0.5")

    overlap_size = int(chunk_size * overlap_ratio)

    if len(text) <= chunk_size:
        return [TextChunk(text=text, start_char=0, end_char=len(text), chunk_index=0, total_chunks=1)]

    chunks: list[TextChunk] = []
    pos = 0

    while pos < len(text):
        end = min(pos + chunk_size, len(text))

        # Try to split at a sentence boundary
        if end < len(text):
            end = _find_boundary(text, pos, end, chunk_size)

        chunks.append(TextChunk(
            text=text[pos:end],
            start_char=pos,
            end_char=end,
            chunk_index=len(chunks),
            total_chunks=0,
        ))

        if end >= len(text):
            break
        pos = max(end - overlap_size, pos + 1)

    for c in chunks:
        c.total_chunks = len(chunks)

    return chunks


def _find_boundary(text: str, start: int, target_end: int, chunk_size: int) -> int:
    """Find sentence or word boundary near target_end."""
    search_start = max(start, target_end - int(chunk_size * 0.2))
    search_text = text[search_start:target_end]

    # Prefer sentence boundary
    matches = list(_SENTENCE_END.finditer(search_text))
    if matches:
        return search_start + matches[-1].end()

    # Fall back to word boundary
    last_space = search_text.rfind(" ")
    if last_space > 0:
        return search_start + last_space + 1

    return target_end
