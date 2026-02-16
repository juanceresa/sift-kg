"""Document reader — dispatches to configured extraction backend.

The reader module is the main entry point for document text extraction.
It creates the appropriate backend (kreuzberg or pdfplumber) and delegates
extraction to it.

Backward-compatible: read_document() still returns a plain string.
"""

import logging
from pathlib import Path

from sift_kg.ingest.base import TextExtractor

logger = logging.getLogger(__name__)


def create_extractor(
    backend: str = "kreuzberg",
    ocr: bool = False,
    ocr_backend: str = "tesseract",
    ocr_language: str = "eng",
) -> TextExtractor:
    """Create a text extractor for the given backend.

    Args:
        backend: Extraction backend name ("kreuzberg" or "pdfplumber")
        ocr: Enable OCR for scanned documents
        ocr_backend: OCR engine (tesseract, easyocr, paddleocr, gcv)
        ocr_language: OCR language code (ISO 639-3)

    Returns:
        TextExtractor instance

    Raises:
        ValueError: If backend name is unknown
    """
    if backend == "kreuzberg":
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        return KreuzbergExtractor(ocr=ocr, ocr_backend=ocr_backend, ocr_language=ocr_language)
    elif backend == "pdfplumber":
        from sift_kg.ingest.pdfplumber_extractor import PdfPlumberExtractor

        return PdfPlumberExtractor(ocr=ocr)
    else:
        raise ValueError(
            f"Unknown extraction backend: {backend!r}. Choose 'kreuzberg' or 'pdfplumber'."
        )


def read_document(
    path: Path,
    ocr: bool = False,
    backend: str = "kreuzberg",
    ocr_backend: str = "tesseract",
    ocr_language: str = "eng",
) -> str:
    """Read text content from a document file.

    Backward-compatible wrapper that returns plain text (string).
    Creates the configured extractor and returns its content.

    Args:
        path: Path to the document file
        ocr: If True, enable OCR for scanned documents
        backend: Extraction backend ("kreuzberg" or "pdfplumber")
        ocr_backend: OCR engine when ocr=True
        ocr_language: OCR language code

    Returns:
        Extracted text content
    """
    extractor = create_extractor(
        backend=backend, ocr=ocr, ocr_backend=ocr_backend, ocr_language=ocr_language
    )
    result = extractor.extract(path)
    return result.content



def discover_documents(directory: Path, backend: str = "kreuzberg") -> list[Path]:
    """Find all supported documents in a directory (recursive).

    Uses a single directory traversal regardless of how many extensions
    are supported (important for kreuzberg's 40+ extensions).

    Args:
        directory: Root directory to search
        backend: Extraction backend (determines supported extensions)

    Returns:
        Sorted list of document paths
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    extractor = create_extractor(backend=backend, ocr=False)
    extensions = extractor.supported_extensions()

    # Single traversal — O(tree_size), not O(tree_size * num_extensions)
    docs = [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    return sorted(docs)
