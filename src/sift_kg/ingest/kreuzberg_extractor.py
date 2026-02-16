"""Kreuzberg extraction backend — local-first text extraction with OCR support.

Uses Kreuzberg for 75+ format support, local OCR (Tesseract/EasyOCR/PaddleOCR),
page-level extraction, and document metadata. Requires kreuzberg>=4.0.0.
"""

import logging
from pathlib import Path

from sift_kg.ingest.base import (
    DocumentMetadata,
    ExtractorResult,
    PageContent,
    format_pages_as_content,
)

logger = logging.getLogger(__name__)

# Kreuzberg-supported extensions (subset of 75+ formats, listing the common ones).
# Kreuzberg auto-detects format, so this is used for discover_documents filtering.
_SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".doc", ".odt", ".rtf",
    # Spreadsheets
    ".xlsx", ".xlsm", ".xls", ".ods", ".csv",
    # Presentations
    ".pptx",
    # Web / data
    ".html", ".htm", ".xml", ".json", ".yaml", ".yml",
    # Text
    ".txt", ".md", ".markdown", ".rst", ".log",
    # eBooks
    ".epub", ".fb2",
    # Email
    ".eml", ".msg",
    # Images (OCR)
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
    # Archives (Kreuzberg can extract text from within)
    ".zip",
    # Academic
    ".bib", ".tex", ".ipynb",
}

# Threshold for near-empty text detection (matches pdfplumber_extractor).
_NEAR_EMPTY_CHARS = 50


def _extract_file(path: Path, config: object) -> object:
    """Call kreuzberg.extract_file_sync. Separate function for testability."""
    from kreuzberg import extract_file_sync

    return extract_file_sync(path, config=config)


def _build_extraction_config(
    ocr: bool, ocr_backend: str, ocr_language: str
) -> object:
    """Build kreuzberg ExtractionConfig. Separate function for testability."""
    from kreuzberg import ExtractionConfig, OcrConfig, PageConfig

    ocr_config = None
    if ocr and ocr_backend != "gcv":
        ocr_config = OcrConfig(
            backend=ocr_backend,
            language=ocr_language,
        )

    return ExtractionConfig(
        ocr=ocr_config,
        force_ocr=False,
        pages=PageConfig(extract_pages=True),
    )


class KreuzbergExtractor:
    """Text extraction using Kreuzberg.

    Supports 75+ formats with optional local OCR (Tesseract, EasyOCR, PaddleOCR).
    Page-level extraction provides [PAGE N] markers in the content string.

    When ocr_backend='gcv', Kreuzberg handles text extraction but Google Cloud
    Vision OCR (via ingest/ocr.py) is used as fallback for near-empty PDFs.
    """

    def __init__(
        self,
        ocr: bool = False,
        ocr_backend: str = "tesseract",
        ocr_language: str = "eng",
    ) -> None:
        self._ocr = ocr
        self._ocr_backend = ocr_backend
        self._ocr_language = ocr_language

    def extract(self, path: Path) -> ExtractorResult:
        """Extract text from a document using Kreuzberg.

        Args:
            path: Path to the document file

        Returns:
            ExtractorResult with text, optional page markers, and metadata
        """
        config = _build_extraction_config(
            self._ocr, self._ocr_backend, self._ocr_language
        )

        try:
            result = _extract_file(path, config)
        except Exception as e:
            logger.error(f"Kreuzberg extraction failed for {path.name}: {e}")
            raise ValueError(f"Failed to extract {path.name}: {e}") from e

        # GCV OCR fallback: if ocr_backend='gcv' and text is near-empty on a PDF,
        # route to existing Google Cloud Vision OCR module
        if (
            self._ocr
            and self._ocr_backend == "gcv"
            and path.suffix.lower() == ".pdf"
            and len(result.content.strip()) < _NEAR_EMPTY_CHARS
        ):
            from sift_kg.ingest.ocr import ocr_pdf

            logger.info(f"Near-empty text from Kreuzberg for {path.name} — falling back to GCV OCR")
            gcv_text = ocr_pdf(path)
            return ExtractorResult(
                content=gcv_text,
                metadata=DocumentMetadata(mime_type="application/pdf"),
            )

        # Build page-level content if available
        # Kreuzberg returns pages as dicts: {page_number, content, tables, images, is_blank}
        pages: list[PageContent] | None = None
        if result.pages:
            pages = [
                PageContent(page_number=p["page_number"], text=p["content"])
                for p in result.pages
            ]

        # Use page-marked content if pages available, with fallback to raw content
        if pages:
            page_content = format_pages_as_content(pages)
            content = page_content if page_content else result.content
        else:
            content = result.content

        # Extract metadata from Kreuzberg's TypedDict
        metadata = DocumentMetadata(mime_type=result.mime_type)
        if isinstance(result.metadata, dict):
            metadata.title = result.metadata.get("title")
            authors = result.metadata.get("authors")
            if isinstance(authors, list) and authors:
                metadata.author = authors[0]
            metadata.date = result.metadata.get("created_at")

        return ExtractorResult(content=content, pages=pages, metadata=metadata)

    def supported_extensions(self) -> set[str]:
        return set(_SUPPORTED_EXTENSIONS)
