"""Base types for pluggable text extraction backends.

Defines the TextExtractor protocol and ExtractorResult data model used by all
extraction backends (pdfplumber, kreuzberg).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class PageContent:
    """Text content from a single page of a document."""

    page_number: int
    text: str


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document (title, author, etc.)."""

    title: str | None = None
    author: str | None = None
    date: str | None = None
    mime_type: str = ""


@dataclass
class ExtractorResult:
    """Result from a text extraction backend.

    The `content` field is the full extracted text (with optional [PAGE N] markers).
    This is what the chunker receives — downstream pipeline is unchanged.
    """

    content: str
    pages: list[PageContent] | None = None
    metadata: DocumentMetadata | None = None


class TextExtractor(Protocol):
    """Protocol for text extraction backends.

    Implementations must provide:
    - extract(path) -> ExtractorResult
    - supported_extensions() -> set of file extensions (e.g. {".pdf", ".docx"})
    """

    def extract(self, path: Path) -> ExtractorResult: ...

    def supported_extensions(self) -> set[str]: ...


def format_pages_as_content(pages: list[PageContent]) -> str:
    """Format page-level text into a single content string with [PAGE N] markers.

    Args:
        pages: List of PageContent objects with page_number and text

    Returns:
        Single string with [PAGE N] markers between page boundaries
    """
    parts = []
    for page in pages:
        if page.text.strip():
            parts.append(f"[PAGE {page.page_number}]\n{page.text}")
    return "\n\n".join(parts)
