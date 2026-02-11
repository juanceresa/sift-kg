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
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            config.validate_api_keys("openai/gpt-4o-mini")

    def test_validate_anthropic_key_present(self):
        """No error when Anthropic key is set and model is Anthropic."""
        config = SiftConfig(anthropic_api_key="sk-ant-test", _env_file=None)
        config.validate_api_keys("anthropic/claude-3-haiku")  # Should not raise

    def test_validate_anthropic_key_missing(self):
        """Error when using Anthropic model without API key."""
        config = SiftConfig(anthropic_api_key=None, _env_file=None)
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
