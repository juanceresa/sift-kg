"""OCR support for scanned PDFs using Google Cloud Vision.

Converts PDF pages to images via PyMuPDF, sends to Vision API for text detection,
and normalizes the resulting text. Requires the [ocr] optional extra:

    pip install sift-kg[ocr]

Authentication uses standard Google Cloud credentials — set GOOGLE_APPLICATION_CREDENTIALS
env var or use Application Default Credentials (gcloud auth application-default login).
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def ocr_pdf(path: Path) -> str:
    """Extract text from a scanned PDF using Google Cloud Vision OCR.

    Opens the PDF with PyMuPDF, renders each page at 300 DPI as PNG,
    sends to Vision API's document_text_detection, and concatenates results.

    Args:
        path: Path to the PDF file

    Returns:
        Extracted and normalized text from all pages

    Raises:
        ImportError: If google-cloud-vision or pymupdf are not installed
    """
    try:
        import pymupdf  # noqa: F811
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for OCR support.\n"
            "Install with: pip install sift-kg[ocr]"
        ) from None

    try:
        from google.cloud import vision
    except ImportError:
        raise ImportError(
            "google-cloud-vision is required for OCR support.\n"
            "Install with: pip install sift-kg[ocr]"
        ) from None

    client = vision.ImageAnnotatorClient()
    doc = pymupdf.open(str(path))
    pages_text: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 300 DPI for good OCR quality
        pix = page.get_pixmap(dpi=300)
        image_bytes = pix.tobytes("png")

        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)

        if response.error.message:
            logger.warning(f"Vision API error on page {page_num + 1}: {response.error.message}")
            continue

        if response.full_text_annotation:
            pages_text.append(response.full_text_annotation.text)
        else:
            logger.debug(f"No text detected on page {page_num + 1}")

    doc.close()

    raw_text = "\n\n".join(pages_text)
    return normalize_ocr_text(raw_text)


def normalize_ocr_text(text: str) -> str:
    """Clean up common OCR artifacts in extracted text.

    - Joins hyphenated line breaks (e.g. "docu-\\nment" -> "document")
    - Collapses excessive blank lines to double newlines
    - Joins mid-sentence line breaks (lowercase after newline)

    Args:
        text: Raw OCR text

    Returns:
        Cleaned text
    """
    # Join hyphenated line breaks: "docu-\nment" -> "document"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Join mid-sentence line breaks (single newline followed by lowercase letter)
    # Uses negative lookbehind to avoid joining after paragraph breaks (\n\n)
    text = re.sub(r"(?<!\n)\n([a-z])", r" \1", text)

    return text.strip()
