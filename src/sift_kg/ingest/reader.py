"""Document reader — extracts text from PDF, DOCX, plain text, and HTML files."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


def read_document(path: Path, ocr: bool = False) -> str:
    """Read text content from a document file.

    Supports PDF (text-searchable or scanned with --ocr), plain text (.txt, .md), and HTML.

    Args:
        path: Path to the document file
        ocr: If True, use Google Cloud Vision OCR for PDF files

    Returns:
        Extracted text content

    Raises:
        ValueError: If file type is unsupported or file can't be read
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix == ".pdf":
        return _read_pdf(path, ocr=ocr)
    elif suffix == ".docx":
        return _read_docx(path)
    elif suffix in {".html", ".htm"}:
        return _read_html(path)
    else:
        return _read_text(path)


# Average chars per page below which a page is considered near-empty.
# Scanned PDFs typically return 0-5 chars/page from pdfplumber.
# A short but real document (e.g. a one-page letter) returns 200+.
_NEAR_EMPTY_THRESHOLD = 15


def _read_pdf(path: Path, ocr: bool = False) -> str:
    """Extract text from a PDF.

    When ocr=True, tries pdfplumber first. If pages are near-empty (likely
    scanned), falls back to Google Cloud Vision OCR automatically. Text-rich
    PDFs skip OCR entirely — no wasted API calls.

    When ocr=False, uses pdfplumber only and warns if text looks thin.
    """
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)

    full_text = "\n\n".join(pages)
    num_pages = len(pages)

    # Check if pdfplumber got near-empty results (likely scanned).
    # num_pages == 0 means pdfplumber couldn't parse the PDF at all — also a scan signal.
    avg_chars = len(full_text.strip()) / num_pages if num_pages > 0 else 0
    is_near_empty = num_pages == 0 or avg_chars < _NEAR_EMPTY_THRESHOLD

    if is_near_empty and ocr:
        from sift_kg.ingest.ocr import ocr_pdf

        logger.info(f"Near-empty text from pdfplumber for {path.name} — falling back to OCR")
        return ocr_pdf(path)

    if is_near_empty and not ocr:
        logger.warning(
            f"Near-empty text extracted from {path.name} — this may be a scanned PDF. "
            "Try: sift extract --ocr"
        )

    return full_text


def _read_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    from docx import Document

    doc = Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _read_text(path: Path) -> str:
    """Read plain text file with encoding fallback."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {path.name}, trying latin-1")
        return path.read_text(encoding="latin-1")


def _read_html(path: Path) -> str:
    """Extract visible text from HTML using BeautifulSoup."""
    from bs4 import BeautifulSoup

    html = _read_text(path)
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def discover_documents(directory: Path) -> list[Path]:
    """Find all supported documents in a directory (recursive).

    Args:
        directory: Root directory to search

    Returns:
        Sorted list of document paths
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    docs = []
    for ext in SUPPORTED_EXTENSIONS:
        docs.extend(directory.rglob(f"*{ext}"))

    return sorted(docs)
