"""Document reader — extracts text from PDF, DOCX, plain text, and HTML files."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


def read_document(path: Path) -> str:
    """Read text content from a document file.

    Supports PDF (text-searchable), plain text (.txt, .md), and HTML.

    Args:
        path: Path to the document file

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
        return _read_pdf(path)
    elif suffix == ".docx":
        return _read_docx(path)
    elif suffix in {".html", ".htm"}:
        return _read_html(path)
    else:
        return _read_text(path)


def _read_pdf(path: Path) -> str:
    """Extract text from a text-searchable PDF using pdfplumber."""
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)

    return "\n\n".join(pages)


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
