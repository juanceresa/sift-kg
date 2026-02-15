"""Tests for sift_kg.ingest (reader, chunker, and OCR)."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sift_kg.ingest.chunker import chunk_text
from sift_kg.ingest.ocr import normalize_ocr_text
from sift_kg.ingest.reader import discover_documents, read_document


class TestReader:
    """Test document reading."""

    def test_read_text_file(self, tmp_dir):
        """Read plain text file."""
        f = tmp_dir / "doc.txt"
        f.write_text("Hello world", encoding="utf-8")
        text = read_document(f)
        assert text == "Hello world"

    def test_read_markdown_file(self, tmp_dir):
        """Read markdown file."""
        f = tmp_dir / "doc.md"
        f.write_text("# Title\n\nParagraph text.", encoding="utf-8")
        text = read_document(f)
        assert "Title" in text
        assert "Paragraph" in text

    def test_read_html_file(self, tmp_dir):
        """Read HTML file extracts text content."""
        f = tmp_dir / "doc.html"
        f.write_text(
            "<html><body><h1>Title</h1><p>Content here.</p></body></html>",
            encoding="utf-8",
        )
        text = read_document(f)
        assert "Title" in text
        assert "Content here" in text
        # Should strip HTML tags
        assert "<h1>" not in text

    def test_read_empty_file(self, tmp_dir):
        """Reading empty file returns empty string."""
        f = tmp_dir / "empty.txt"
        f.write_text("", encoding="utf-8")
        text = read_document(f)
        assert text == ""

    def test_read_unsupported_format(self, tmp_dir):
        """Unsupported file format raises error."""
        f = tmp_dir / "doc.xyz"
        f.write_text("data", encoding="utf-8")
        with pytest.raises((ValueError, Exception)):
            read_document(f)

    def test_read_nonexistent_file(self, tmp_dir):
        """Missing file raises error."""
        with pytest.raises((FileNotFoundError, Exception)):
            read_document(tmp_dir / "nonexistent.txt")

    def test_read_latin1_fallback(self, tmp_dir):
        """Text file with Latin-1 encoding is handled via fallback."""
        f = tmp_dir / "latin.txt"
        # Write bytes that are valid Latin-1 but not UTF-8
        f.write_bytes(b"Caf\xe9 au lait")
        text = read_document(f)
        assert "Caf" in text


class TestDiscoverDocuments:
    """Test document discovery."""

    def test_discover_supported_files(self, tmp_dir):
        """Finds txt, md, html files."""
        (tmp_dir / "a.txt").write_text("a")
        (tmp_dir / "b.md").write_text("b")
        (tmp_dir / "c.html").write_text("c")
        (tmp_dir / "d.xyz").write_text("d")  # unsupported

        docs = discover_documents(tmp_dir)
        extensions = {d.suffix for d in docs}
        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".html" in extensions
        assert ".xyz" not in extensions

    def test_discover_recursive(self, tmp_dir):
        """Finds files in subdirectories."""
        sub = tmp_dir / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("content")

        docs = discover_documents(tmp_dir)
        assert any("nested.txt" in str(d) for d in docs)

    def test_discover_empty_dir(self, tmp_dir):
        """Empty directory returns empty list."""
        docs = discover_documents(tmp_dir)
        assert docs == []

    def test_discover_sorted(self, tmp_dir):
        """Results are sorted."""
        (tmp_dir / "c.txt").write_text("c")
        (tmp_dir / "a.txt").write_text("a")
        (tmp_dir / "b.txt").write_text("b")

        docs = discover_documents(tmp_dir)
        names = [d.name for d in docs]
        assert names == sorted(names)


class TestChunker:
    """Test text chunking."""

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size returns one chunk."""
        text = "Short text."
        chunks = chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0

    def test_long_text_multiple_chunks(self):
        """Long text is split into multiple chunks."""
        text = "Word " * 500  # ~2500 chars
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) > 1

    def test_chunk_overlap(self):
        """Adjacent chunks have overlapping content."""
        sentences = ". ".join(f"Sentence number {i}" for i in range(50))
        chunks = chunk_text(sentences, chunk_size=200, overlap_ratio=0.2)

        if len(chunks) >= 2:
            # Check that the end of chunk 0 overlaps with start of chunk 1
            chunk0_end = chunks[0].text[-50:]
            chunk1_start = chunks[1].text[:50]
            # At least some content should be shared
            chunk0_words = set(chunk0_end.split())
            chunk1_words = set(chunk1_start.split())
            assert chunk0_words & chunk1_words  # non-empty intersection

    def test_chunk_indices_sequential(self):
        """Chunk indices are 0, 1, 2, ..."""
        text = "Word " * 500
        chunks = chunk_text(text, chunk_size=200)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_text(self):
        """Empty text returns empty or single empty chunk."""
        chunks = chunk_text("")
        # Either empty list or single chunk with empty text
        assert len(chunks) <= 1

    def test_chunk_preserves_all_content(self):
        """All original text appears in at least one chunk."""
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=200)

        all_chunk_text = " ".join(c.text for c in chunks)
        for word in words:
            assert word in all_chunk_text

    def test_chunk_dataclass_fields(self):
        """TextChunk has expected fields."""
        chunks = chunk_text("Hello world.", chunk_size=1000)
        chunk = chunks[0]
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "chunk_index")
        assert isinstance(chunk.text, str)
        assert isinstance(chunk.chunk_index, int)


class TestNormalizeOcrText:
    """Test OCR text normalization."""

    def test_joins_hyphenated_line_breaks(self):
        assert normalize_ocr_text("docu-\nment") == "document"

    def test_collapses_multiple_newlines(self):
        result = normalize_ocr_text("paragraph one\n\n\n\n\nparagraph two")
        assert result == "paragraph one\n\nparagraph two"

    def test_joins_mid_sentence_line_breaks(self):
        result = normalize_ocr_text("the quick brown\nfox jumps over")
        assert result == "the quick brown fox jumps over"

    def test_preserves_sentence_boundaries(self):
        result = normalize_ocr_text("First sentence.\nSecond sentence.")
        # Capital S means new sentence — should NOT be joined
        assert "Second" in result
        assert "\nSecond" in result

    def test_empty_string(self):
        assert normalize_ocr_text("") == ""

    def test_combined_artifacts(self):
        text = "The docu-\nment was impor-\ntant.\n\n\n\nIt contained\nevidence."
        result = normalize_ocr_text(text)
        assert "document" in result
        assert "important" in result
        assert "\n\n\n" not in result
        assert "contained evidence" in result


class TestOcrIntegration:
    """Test OCR routing and error handling."""

    def test_ocr_autodetect_falls_back_on_near_empty(self, tmp_dir):
        """ocr=True with near-empty pdfplumber result falls back to OCR."""
        pdf_path = tmp_dir / "scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""  # scanned — empty

        with patch("pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = [mock_page, mock_page]
            mock_open.return_value = mock_pdf

            with patch("sift_kg.ingest.ocr.ocr_pdf", return_value="OCR text") as mock_ocr:
                text = read_document(pdf_path, ocr=True)

        mock_ocr.assert_called_once_with(pdf_path)
        assert text == "OCR text"

    def test_ocr_skips_text_rich_pdf(self, tmp_dir):
        """ocr=True with text-rich PDF skips OCR, uses pdfplumber result."""
        pdf_path = tmp_dir / "normal.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "A" * 500  # plenty of text

        with patch("pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = [mock_page]
            mock_open.return_value = mock_pdf

            with patch("sift_kg.ingest.ocr.ocr_pdf") as mock_ocr:
                text = read_document(pdf_path, ocr=True)

        mock_ocr.assert_not_called()
        assert "A" * 500 in text

    def test_ocr_false_never_calls_ocr(self, tmp_dir):
        """ocr=False never calls OCR even on near-empty PDF."""
        pdf_path = tmp_dir / "scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""

        with patch("pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = [mock_page]
            mock_open.return_value = mock_pdf

            with patch("sift_kg.ingest.ocr.ocr_pdf") as mock_ocr:
                read_document(pdf_path, ocr=False)

        mock_ocr.assert_not_called()

    def test_ocr_import_error_pymupdf(self):
        """Clear error message when pymupdf is missing."""
        import importlib

        from sift_kg.ingest import ocr as ocr_module

        with patch.dict("sys.modules", {"pymupdf": None}):
            with patch("builtins.__import__", side_effect=_import_blocker("pymupdf")):
                importlib.reload(ocr_module)

                with pytest.raises(ImportError, match="PyMuPDF is required"):
                    ocr_module.ocr_pdf(Path("/fake/doc.pdf"))

    def test_near_empty_warning_without_ocr(self, tmp_dir, caplog):
        """Near-empty text from pdfplumber triggers a warning when ocr=False."""
        pdf_path = tmp_dir / "thin.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""  # near-empty

        with patch("pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = [mock_page, mock_page, mock_page]
            mock_open.return_value = mock_pdf

            with caplog.at_level(logging.WARNING):
                read_document(pdf_path, ocr=False)

        assert "scanned PDF" in caplog.text
        assert "--ocr" in caplog.text


def _import_blocker(blocked_module: str):
    """Create an import side_effect that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _blocked_import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"Mocked: {name} not installed")
        return real_import(name, *args, **kwargs)

    return _blocked_import
