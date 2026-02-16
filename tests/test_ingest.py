"""Tests for sift_kg.ingest (reader, chunker, and OCR)."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sift_kg.ingest.base import (
    DocumentMetadata,
    ExtractorResult,
    PageContent,
    TextExtractor,
    format_pages_as_content,
)
from sift_kg.ingest.chunker import chunk_text
from sift_kg.ingest.ocr import normalize_ocr_text
from sift_kg.ingest.reader import discover_documents, read_document


class TestExtractorResult:
    """Test ExtractorResult data model."""

    def test_basic_result(self):
        """ExtractorResult holds content string."""
        result = ExtractorResult(content="Hello world")
        assert result.content == "Hello world"
        assert result.pages is None
        assert result.metadata is None

    def test_result_with_pages(self):
        """ExtractorResult can hold page-level content."""
        pages = [
            PageContent(page_number=1, text="Page one text"),
            PageContent(page_number=2, text="Page two text"),
        ]
        result = ExtractorResult(content="Full text", pages=pages)
        assert len(result.pages) == 2
        assert result.pages[0].page_number == 1
        assert result.pages[1].text == "Page two text"

    def test_result_with_metadata(self):
        """ExtractorResult can hold document metadata."""
        meta = DocumentMetadata(title="My Doc", author="Alice", mime_type="application/pdf")
        result = ExtractorResult(content="text", metadata=meta)
        assert result.metadata.title == "My Doc"
        assert result.metadata.author == "Alice"
        assert result.metadata.date is None

    def test_content_with_page_markers(self):
        """Content string can contain page markers."""
        content = "[PAGE 1]\nFirst page.\n\n[PAGE 2]\nSecond page."
        result = ExtractorResult(content=content)
        assert "[PAGE 1]" in result.content
        assert "[PAGE 2]" in result.content


class TestFormatPagesAsContent:
    """Test page marker formatting."""

    def test_formats_pages_with_markers(self):
        pages = [
            PageContent(page_number=1, text="First page text."),
            PageContent(page_number=2, text="Second page text."),
        ]
        result = format_pages_as_content(pages)
        assert result == "[PAGE 1]\nFirst page text.\n\n[PAGE 2]\nSecond page text."

    def test_skips_empty_pages(self):
        pages = [
            PageContent(page_number=1, text="Content"),
            PageContent(page_number=2, text="   "),
            PageContent(page_number=3, text="More content"),
        ]
        result = format_pages_as_content(pages)
        assert "[PAGE 2]" not in result
        assert "[PAGE 1]" in result
        assert "[PAGE 3]" in result

    def test_empty_pages_list(self):
        result = format_pages_as_content([])
        assert result == ""

    def test_single_page(self):
        pages = [PageContent(page_number=1, text="Only page.")]
        result = format_pages_as_content(pages)
        assert result == "[PAGE 1]\nOnly page."


class TestPdfPlumberExtractor:
    """Test the pdfplumber extraction backend."""

    def test_extract_text_file(self, tmp_dir):
        """Extracts plain text files."""
        from sift_kg.ingest.pdfplumber_extractor import PdfPlumberExtractor

        f = tmp_dir / "doc.txt"
        f.write_text("Hello world", encoding="utf-8")
        extractor = PdfPlumberExtractor(ocr=False)
        result = extractor.extract(f)
        assert result.content == "Hello world"
        assert result.pages is None
        assert result.metadata is not None
        assert result.metadata.mime_type == "text/plain"

    def test_extract_html_file(self, tmp_dir):
        """Extracts HTML files, stripping tags."""
        from sift_kg.ingest.pdfplumber_extractor import PdfPlumberExtractor

        f = tmp_dir / "doc.html"
        f.write_text("<html><body><h1>Title</h1><p>Content</p></body></html>")
        extractor = PdfPlumberExtractor(ocr=False)
        result = extractor.extract(f)
        assert "Title" in result.content
        assert "Content" in result.content
        assert "<h1>" not in result.content

    def test_extract_markdown_file(self, tmp_dir):
        """Extracts markdown files."""
        from sift_kg.ingest.pdfplumber_extractor import PdfPlumberExtractor

        f = tmp_dir / "doc.md"
        f.write_text("# Title\n\nParagraph text.")
        extractor = PdfPlumberExtractor(ocr=False)
        result = extractor.extract(f)
        assert "Title" in result.content

    def test_supported_extensions(self):
        """Returns correct set of supported extensions."""
        from sift_kg.ingest.pdfplumber_extractor import PdfPlumberExtractor

        extractor = PdfPlumberExtractor(ocr=False)
        exts = extractor.supported_extensions()
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".txt" in exts
        assert ".md" in exts
        assert ".html" in exts
        assert ".htm" in exts

    def test_unsupported_extension_raises(self, tmp_dir):
        """Unsupported file type raises ValueError."""
        from sift_kg.ingest.pdfplumber_extractor import PdfPlumberExtractor

        f = tmp_dir / "doc.xyz"
        f.write_text("data")
        extractor = PdfPlumberExtractor(ocr=False)
        with pytest.raises(ValueError, match="Unsupported"):
            extractor.extract(f)

    def test_pdf_ocr_fallback(self, tmp_dir):
        """OCR fallback triggers on near-empty PDF when ocr=True."""
        from sift_kg.ingest.pdfplumber_extractor import PdfPlumberExtractor

        pdf_path = tmp_dir / "scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""

        with patch("pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = [mock_page, mock_page]
            mock_open.return_value = mock_pdf

            with patch("sift_kg.ingest.ocr.ocr_pdf", return_value="OCR text") as mock_ocr:
                extractor = PdfPlumberExtractor(ocr=True)
                result = extractor.extract(pdf_path)

        mock_ocr.assert_called_once_with(pdf_path)
        assert result.content == "OCR text"


class TestCreateExtractor:
    """Test extractor factory function."""

    def test_create_pdfplumber_extractor(self):
        """Creates PdfPlumberExtractor for 'pdfplumber' backend."""
        from sift_kg.ingest.pdfplumber_extractor import PdfPlumberExtractor
        from sift_kg.ingest.reader import create_extractor

        extractor = create_extractor(backend="pdfplumber", ocr=False)
        assert isinstance(extractor, PdfPlumberExtractor)

    def test_create_kreuzberg_extractor(self):
        """Creates KreuzbergExtractor for 'kreuzberg' backend."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor
        from sift_kg.ingest.reader import create_extractor

        extractor = create_extractor(backend="kreuzberg", ocr=False)
        assert isinstance(extractor, KreuzbergExtractor)

    def test_invalid_backend_raises(self):
        """Unknown backend name raises ValueError."""
        from sift_kg.ingest.reader import create_extractor

        with pytest.raises(ValueError, match="Unknown extraction backend"):
            create_extractor(backend="nonexistent", ocr=False)


class TestReader:
    """Test document reading via pdfplumber backend."""

    def test_read_text_file(self, tmp_dir):
        """Read plain text file."""
        f = tmp_dir / "doc.txt"
        f.write_text("Hello world", encoding="utf-8")
        text = read_document(f, backend="pdfplumber")
        assert text == "Hello world"

    def test_read_markdown_file(self, tmp_dir):
        """Read markdown file."""
        f = tmp_dir / "doc.md"
        f.write_text("# Title\n\nParagraph text.", encoding="utf-8")
        text = read_document(f, backend="pdfplumber")
        assert "Title" in text
        assert "Paragraph" in text

    def test_read_html_file(self, tmp_dir):
        """Read HTML file extracts text content."""
        f = tmp_dir / "doc.html"
        f.write_text(
            "<html><body><h1>Title</h1><p>Content here.</p></body></html>",
            encoding="utf-8",
        )
        text = read_document(f, backend="pdfplumber")
        assert "Title" in text
        assert "Content here" in text
        # Should strip HTML tags
        assert "<h1>" not in text

    def test_read_empty_file(self, tmp_dir):
        """Reading empty file returns empty string."""
        f = tmp_dir / "empty.txt"
        f.write_text("", encoding="utf-8")
        text = read_document(f, backend="pdfplumber")
        assert text == ""

    def test_read_unsupported_format(self, tmp_dir):
        """Unsupported file format raises error."""
        f = tmp_dir / "doc.xyz"
        f.write_text("data", encoding="utf-8")
        with pytest.raises((ValueError, Exception)):
            read_document(f, backend="pdfplumber")

    def test_read_nonexistent_file(self, tmp_dir):
        """Missing file raises error."""
        with pytest.raises((FileNotFoundError, Exception)):
            read_document(tmp_dir / "nonexistent.txt", backend="pdfplumber")

    def test_read_latin1_fallback(self, tmp_dir):
        """Text file with Latin-1 encoding is handled via fallback."""
        f = tmp_dir / "latin.txt"
        # Write bytes that are valid Latin-1 but not UTF-8
        f.write_bytes(b"Caf\xe9 au lait")
        text = read_document(f, backend="pdfplumber")
        assert "Caf" in text


class TestDiscoverDocuments:
    """Test document discovery (pinned to pdfplumber backend)."""

    def test_discover_supported_files(self, tmp_dir):
        """Finds txt, md, html files."""
        (tmp_dir / "a.txt").write_text("a")
        (tmp_dir / "b.md").write_text("b")
        (tmp_dir / "c.html").write_text("c")
        (tmp_dir / "d.xyz").write_text("d")  # unsupported

        docs = discover_documents(tmp_dir, backend="pdfplumber")
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

        docs = discover_documents(tmp_dir, backend="pdfplumber")
        assert any("nested.txt" in str(d) for d in docs)

    def test_discover_empty_dir(self, tmp_dir):
        """Empty directory returns empty list."""
        docs = discover_documents(tmp_dir, backend="pdfplumber")
        assert docs == []

    def test_discover_sorted(self, tmp_dir):
        """Results are sorted."""
        (tmp_dir / "c.txt").write_text("c")
        (tmp_dir / "a.txt").write_text("a")
        (tmp_dir / "b.txt").write_text("b")

        docs = discover_documents(tmp_dir, backend="pdfplumber")
        names = [d.name for d in docs]
        assert names == sorted(names)


class TestDiscoverDocumentsKreuzberg:
    """Test document discovery with kreuzberg backend (broader format support)."""

    def test_discovers_kreuzberg_formats(self, tmp_dir):
        """Kreuzberg backend discovers formats beyond pdfplumber's set."""
        (tmp_dir / "a.txt").write_text("text")
        (tmp_dir / "b.csv").write_text("a,b\n1,2")
        (tmp_dir / "c.json").write_text('{"key": "val"}')
        (tmp_dir / "d.xml").write_text("<root/>")
        (tmp_dir / "e.xyz").write_text("unsupported")

        docs = discover_documents(tmp_dir, backend="kreuzberg")
        extensions = {d.suffix for d in docs}
        assert ".txt" in extensions
        assert ".csv" in extensions
        assert ".json" in extensions
        assert ".xyz" not in extensions

    def test_kreuzberg_superset_of_pdfplumber(self, tmp_dir):
        """Kreuzberg discovers everything pdfplumber does, plus more."""
        (tmp_dir / "a.txt").write_text("text")
        (tmp_dir / "b.md").write_text("# heading")
        (tmp_dir / "c.html").write_text("<p>hi</p>")
        (tmp_dir / "d.csv").write_text("a,b")

        pdfplumber_docs = set(d.name for d in discover_documents(tmp_dir, backend="pdfplumber"))
        kreuzberg_docs = set(d.name for d in discover_documents(tmp_dir, backend="kreuzberg"))
        assert pdfplumber_docs.issubset(kreuzberg_docs)
        assert len(kreuzberg_docs) > len(pdfplumber_docs)  # csv only in kreuzberg


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
                text = read_document(pdf_path, ocr=True, backend="pdfplumber")

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
                text = read_document(pdf_path, ocr=True, backend="pdfplumber")

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
                read_document(pdf_path, ocr=False, backend="pdfplumber")

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
                read_document(pdf_path, ocr=False, backend="pdfplumber")

        assert "scanned PDF" in caplog.text
        assert "--ocr" in caplog.text


class TestKreuzbergExtractor:
    """Test the Kreuzberg extraction backend against the real library.

    All tests use real kreuzberg calls except GCV OCR (needs credentials)
    and the empty-pages edge case (impossible to produce with real files).
    """

    def test_extract_text_file(self, tmp_dir):
        """Extracts plain text."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        f = tmp_dir / "doc.txt"
        f.write_text("Hello from Kreuzberg", encoding="utf-8")
        extractor = KreuzbergExtractor(ocr=False)
        result = extractor.extract(f)
        assert "Hello from Kreuzberg" in result.content

    def test_extract_html_file(self, tmp_dir):
        """Extracts HTML, stripping tags."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        f = tmp_dir / "doc.html"
        f.write_text("<html><body><h1>Title</h1><p>Kreuzberg HTML</p></body></html>")
        extractor = KreuzbergExtractor(ocr=False)
        result = extractor.extract(f)
        assert "Kreuzberg HTML" in result.content

    def test_extract_csv_file(self, tmp_dir):
        """Extracts CSV — a format pdfplumber can't handle."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        f = tmp_dir / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25")
        extractor = KreuzbergExtractor(ocr=False)
        result = extractor.extract(f)
        assert "Alice" in result.content
        assert "Bob" in result.content

    def test_extract_json_file(self, tmp_dir):
        """Extracts JSON."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        f = tmp_dir / "data.json"
        f.write_text('{"name": "Alice", "role": "Engineer"}')
        extractor = KreuzbergExtractor(ocr=False)
        result = extractor.extract(f)
        assert "Alice" in result.content

    def test_extract_returns_metadata(self, tmp_dir):
        """Returns metadata with mime_type."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        f = tmp_dir / "doc.txt"
        f.write_text("metadata test")
        extractor = KreuzbergExtractor(ocr=False)
        result = extractor.extract(f)
        assert result.metadata is not None
        assert result.metadata.mime_type == "text/plain"

    def test_metadata_maps_title_from_html(self, tmp_dir):
        """HTML <title> is mapped to DocumentMetadata.title."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        f = tmp_dir / "doc.html"
        f.write_text("<html><head><title>My Report</title></head><body><p>Content</p></body></html>")
        extractor = KreuzbergExtractor(ocr=False)
        result = extractor.extract(f)
        assert result.metadata.title == "My Report"

    def test_supported_extensions_broader(self):
        """Kreuzberg supports more formats than pdfplumber backend."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        extractor = KreuzbergExtractor(ocr=False)
        exts = extractor.supported_extensions()
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".txt" in exts
        assert ".html" in exts
        assert ".xlsx" in exts
        assert ".pptx" in exts
        assert ".epub" in exts
        assert ".csv" in exts
        assert ".json" in exts
        assert ".xml" in exts
        assert ".rtf" in exts

    def test_config_builds_with_real_kreuzberg(self):
        """_build_extraction_config produces a real kreuzberg ExtractionConfig."""
        from kreuzberg import ExtractionConfig

        from sift_kg.ingest.kreuzberg_extractor import _build_extraction_config

        config = _build_extraction_config(ocr=False, ocr_backend="tesseract", ocr_language="eng")
        assert isinstance(config, ExtractionConfig)

    def test_config_includes_ocr_when_enabled(self):
        """OCR config is set when ocr=True with a kreuzberg-native backend."""
        from sift_kg.ingest.kreuzberg_extractor import _build_extraction_config

        config = _build_extraction_config(ocr=True, ocr_backend="tesseract", ocr_language="fra")
        assert config.ocr is not None
        assert config.ocr.backend == "tesseract"
        assert config.ocr.language == "fra"

    def test_config_skips_ocr_for_gcv(self):
        """OCR config is None when ocr_backend='gcv' (handled by GCV fallback)."""
        from sift_kg.ingest.kreuzberg_extractor import _build_extraction_config

        config = _build_extraction_config(ocr=True, ocr_backend="gcv", ocr_language="eng")
        assert config.ocr is None

    def test_page_markers_from_real_pdf(self):
        """Real multi-page PDF produces [PAGE N] markers in content."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        pdf = Path("examples/transformers/docs/02_bert_2018.pdf")
        if not pdf.exists():
            pytest.skip("Example PDF not available")

        extractor = KreuzbergExtractor(ocr=False)
        result = extractor.extract(pdf)

        assert "[PAGE 1]" in result.content
        assert "[PAGE 2]" in result.content
        assert result.pages is not None
        assert len(result.pages) > 1
        assert result.pages[0].page_number == 1
        assert "BERT" in result.pages[0].text

    def test_pdf_metadata(self):
        """Real PDF returns metadata with mime_type."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        pdf = Path("examples/transformers/docs/02_bert_2018.pdf")
        if not pdf.exists():
            pytest.skip("Example PDF not available")

        extractor = KreuzbergExtractor(ocr=False)
        result = extractor.extract(pdf)
        assert result.metadata.mime_type == "application/pdf"

    def test_gcv_ocr_fallback_on_near_empty_pdf(self, tmp_dir):
        """ocr_backend='gcv' routes to GCV OCR for near-empty PDFs.
        Mocked — GCV requires API credentials."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        pdf_path = tmp_dir / "scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_result = MagicMock()
        mock_result.content = ""
        mock_result.pages = None
        mock_result.metadata = {}
        mock_result.mime_type = "application/pdf"

        with patch("sift_kg.ingest.kreuzberg_extractor._extract_file", return_value=mock_result):
            with patch("sift_kg.ingest.ocr.ocr_pdf", return_value="GCV OCR text") as mock_ocr:
                extractor = KreuzbergExtractor(ocr=True, ocr_backend="gcv")
                result = extractor.extract(pdf_path)

        mock_ocr.assert_called_once_with(pdf_path)
        assert result.content == "GCV OCR text"

    def test_gcv_ocr_skips_text_rich_pdf(self):
        """ocr_backend='gcv' does NOT call GCV when Kreuzberg extracts text."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        pdf = Path("examples/transformers/docs/02_bert_2018.pdf")
        if not pdf.exists():
            pytest.skip("Example PDF not available")

        with patch("sift_kg.ingest.ocr.ocr_pdf") as mock_ocr:
            extractor = KreuzbergExtractor(ocr=True, ocr_backend="gcv")
            result = extractor.extract(pdf)

        mock_ocr.assert_not_called()
        assert "BERT" in result.content

    def test_falls_back_to_raw_content_when_pages_empty(self, tmp_dir):
        """If Kreuzberg returns pages but all are empty, uses raw content.
        Mocked — edge case impossible to produce with real files."""
        from sift_kg.ingest.kreuzberg_extractor import KreuzbergExtractor

        f = tmp_dir / "doc.txt"
        f.write_text("raw content here")

        mock_result = MagicMock()
        mock_result.content = "raw content here"
        mock_result.pages = [
            {"page_number": 1, "content": "", "tables": [], "images": [], "is_blank": True},
            {"page_number": 2, "content": "   ", "tables": [], "images": [], "is_blank": True},
        ]
        mock_result.metadata = {}
        mock_result.mime_type = "text/plain"

        with patch("sift_kg.ingest.kreuzberg_extractor._extract_file", return_value=mock_result):
            extractor = KreuzbergExtractor(ocr=False)
            result = extractor.extract(f)

        assert result.content == "raw content here"


def _import_blocker(blocked_module: str):
    """Create an import side_effect that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _blocked_import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"Mocked: {name} not installed")
        return real_import(name, *args, **kwargs)

    return _blocked_import
