"""Tests for sift_kg.config."""

import os
from unittest.mock import patch

import pytest

from sift_kg.config import SiftConfig


class TestSiftConfig:
    """Test configuration loading and validation."""

    def test_default_config_loads(self):
        """Config loads with defaults when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            config = SiftConfig(
                _env_file=None,  # Don't load .env in tests
            )
        assert config.default_model == "openai/gpt-4o-mini"
        assert config.output_dir.name == "output"

    def test_custom_model_from_env(self):
        """Custom model loaded from environment variable."""
        with patch.dict(os.environ, {"SIFT_DEFAULT_MODEL": "anthropic/claude-3-haiku"}):
            config = SiftConfig(_env_file=None)
        assert config.default_model == "anthropic/claude-3-haiku"

    def test_validate_openai_key_present(self):
        """No error when OpenAI key is set and model is OpenAI."""
        config = SiftConfig(openai_api_key="sk-test123", _env_file=None)
        config.validate_api_keys("openai/gpt-4o-mini")  # Should not raise

    def test_validate_openai_key_missing(self):
        """Error when using OpenAI model without API key."""
        config = SiftConfig(openai_api_key=None, _env_file=None)
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                config.validate_api_keys("openai/gpt-4o-mini")

    def test_validate_anthropic_key_present(self):
        """No error when Anthropic key is set and model is Anthropic."""
        config = SiftConfig(anthropic_api_key="sk-ant-test", _env_file=None)
        config.validate_api_keys("anthropic/claude-3-haiku")  # Should not raise

    def test_validate_anthropic_key_missing(self):
        """Error when using Anthropic model without API key."""
        config = SiftConfig(anthropic_api_key=None, _env_file=None)
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                config.validate_api_keys("anthropic/claude-3-haiku")

    def test_ollama_needs_no_key(self):
        """Ollama models don't require API keys."""
        config = SiftConfig(_env_file=None)
        config.validate_api_keys("ollama/llama3")  # Should not raise

    def test_output_dir_created(self, tmp_dir):
        """Output directory is created if it doesn't exist."""
        out = tmp_dir / "custom_output"
        config = SiftConfig(output_dir=out, _env_file=None)
        assert config.output_dir.resolve() == out.resolve()


class TestExtractionConfig:
    """Test extraction backend configuration fields."""

    def test_default_extraction_backend(self):
        """Default extraction backend is kreuzberg."""
        with patch.dict(os.environ, {}, clear=True):
            config = SiftConfig(_env_file=None)
        assert config.extraction_backend == "kreuzberg"

    def test_custom_extraction_backend_from_env(self):
        """Extraction backend can be set via env var."""
        with patch.dict(os.environ, {"SIFT_EXTRACTION_BACKEND": "pdfplumber"}):
            config = SiftConfig(_env_file=None)
        assert config.extraction_backend == "pdfplumber"

    def test_default_ocr_backend(self):
        """Default OCR backend is tesseract."""
        with patch.dict(os.environ, {}, clear=True):
            config = SiftConfig(_env_file=None)
        assert config.ocr_backend == "tesseract"

    def test_custom_ocr_backend_from_env(self):
        """OCR backend can be set via env var."""
        with patch.dict(os.environ, {"SIFT_OCR_BACKEND": "easyocr"}):
            config = SiftConfig(_env_file=None)
        assert config.ocr_backend == "easyocr"

    def test_default_ocr_language(self):
        """Default OCR language is eng."""
        with patch.dict(os.environ, {}, clear=True):
            config = SiftConfig(_env_file=None)
        assert config.ocr_language == "eng"

    def test_invalid_extraction_backend_raises(self):
        """Invalid extraction backend raises ValueError."""
        with pytest.raises(Exception, match="Invalid extraction backend"):
            SiftConfig(extraction_backend="nonexistent", _env_file=None)

    def test_invalid_ocr_backend_raises(self):
        """Invalid OCR backend raises ValueError."""
        with pytest.raises(Exception, match="Invalid OCR backend"):
            SiftConfig(ocr_backend="nonexistent", _env_file=None)

    def test_sift_yaml_extraction_block(self, tmp_dir):
        """sift.yaml extraction block is read correctly."""
        yaml_path = tmp_dir / "sift.yaml"
        yaml_path.write_text(
            "extraction:\n  backend: pdfplumber\n  ocr_backend: paddleocr\n  ocr_language: fra\n"
        )
        from sift_kg.config import _ProjectYamlSource

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            source = _ProjectYamlSource(SiftConfig)
            data = source()
            assert data.get("extraction_backend") == "pdfplumber"
            assert data.get("ocr_backend") == "paddleocr"
            assert data.get("ocr_language") == "fra"
        finally:
            os.chdir(original_cwd)
