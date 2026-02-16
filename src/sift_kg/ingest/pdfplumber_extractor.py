"""PdfPlumber extraction backend — wraps the original reader.py logic.

Supports PDF (via pdfplumber), DOCX (via python-docx), HTML (via BeautifulSoup),
and plain text files. Google Cloud Vision OCR is available as a fallback for
scanned PDFs when ocr=True.
"""

import logging
from pathlib import Path

from sift_kg.ingest.base import DocumentMetadata, ExtractorResult

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}

_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
}

# Average chars per page below which a page is considered near-empty (scanned PDF).
_NEAR_EMPTY_THRESHOLD = 15


class PdfPlumberExtractor:
    """Text extraction using pdfplumber, python-docx, and BeautifulSoup.

    This is the legacy backend. It supports 6 file formats and optional
    Google Cloud Vision OCR for scanned PDFs.
    """

    def __init__(self, ocr: bool = False) -> None:
        self._ocr = ocr

    def extract(self, path: Path) -> ExtractorResult:
        """Extract text from a document file.

        Args:
            path: Path to the document file

        Returns:
            ExtractorResult with extracted text content

        Raises:
            ValueError: If file type is unsupported
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        mime = _MIME_TYPES.get(suffix, "")

        if suffix == ".pdf":
            content = self._read_pdf(path)
        elif suffix == ".docx":
            content = self._read_docx(path)
        elif suffix in {".html", ".htm"}:
            content = self._read_html(path)
        else:
            content = self._read_text(path)

        return ExtractorResult(
            content=content,
            metadata=DocumentMetadata(mime_type=mime),
        )

    def supported_extensions(self) -> set[str]:
        return set(_SUPPORTED_EXTENSIONS)

    def _read_pdf(self, path: Path) -> str:
        """Extract text from PDF, with optional GCV OCR fallback."""
        import pdfplumber

        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)

        full_text = "\n\n".join(pages)
        num_pages = len(pages)

        avg_chars = len(full_text.strip()) / num_pages if num_pages > 0 else 0
        is_near_empty = num_pages == 0 or avg_chars < _NEAR_EMPTY_THRESHOLD

        if is_near_empty and self._ocr:
            from sift_kg.ingest.ocr import ocr_pdf

            logger.info(f"Near-empty text from pdfplumber for {path.name} — falling back to OCR")
            return ocr_pdf(path)

        if is_near_empty and not self._ocr:
            logger.warning(
                f"Near-empty text extracted from {path.name} — this may be a scanned PDF. "
                "Try: sift extract --ocr"
            )

        return full_text

    def _read_docx(self, path: Path) -> str:
        """Extract text from DOCX using python-docx."""
        from docx import Document

        doc = Document(path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _read_text(self, path: Path) -> str:
        """Read plain text with encoding fallback."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed for {path.name}, trying latin-1")
            return path.read_text(encoding="latin-1")

    def _read_html(self, path: Path) -> str:
        """Extract visible text from HTML."""
        from bs4 import BeautifulSoup

        html = self._read_text(path)
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "head"]):
            tag.decompose()

        return soup.get_text(separator="\n", strip=True)
