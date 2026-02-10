"""Configuration management for sift-kg using pydantic-settings.

This module provides the SiftConfig class for managing application settings
from environment variables and .env files. All configuration is type-safe
and validated using Pydantic models.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SiftConfig(BaseSettings):
    """Configuration settings for sift-kg loaded from environment variables.

    Settings are loaded from:
    1. Environment variables (with SIFT_ prefix)
    2. .env file in current directory
    3. Default values defined in field definitions

    All environment variables should be prefixed with SIFT_ (e.g., SIFT_OPENAI_API_KEY).
    Empty string values in environment variables are treated as unset.

    Example:
        >>> config = SiftConfig()
        >>> config.validate_api_keys("openai/gpt-4o-mini")
        >>> print(config.output_dir)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SIFT_",
        extra="ignore",
        env_ignore_empty=True,
    )

    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for GPT models. Get from: https://platform.openai.com/api-keys"
    )

    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude models. Get from: https://console.anthropic.com/settings/keys"
    )

    default_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Default LLM model in format provider/model-name (e.g., openai/gpt-4o-mini, anthropic/claude-haiku, ollama/llama3.3)"
    )

    output_dir: Path = Field(
        default=Path("output"),
        description="Directory for output files (extractions, graph, narrative)"
    )

    domain_path: Optional[Path] = Field(
        default=None,
        description="Path to custom domain YAML file (uses default domain if not set)"
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def resolve_output_dir(cls, v: Path | str) -> Path:
        """Convert output_dir to absolute path and create if missing.

        Args:
            v: Path string or Path object from config

        Returns:
            Resolved absolute Path with directory created
        """
        path = Path(v) if isinstance(v, str) else v
        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def validate_api_keys(self, model: str) -> None:
        """Validate that required API key exists for the specified model provider.

        Args:
            model: Model string in format provider/model-name (e.g., openai/gpt-4o-mini)

        Raises:
            ValueError: If required API key is missing for the model provider

        Example:
            >>> config = SiftConfig(openai_api_key="sk-...")
            >>> config.validate_api_keys("openai/gpt-4o-mini")  # OK
            >>> config.validate_api_keys("anthropic/claude-haiku")  # Raises ValueError
        """
        if model.startswith("openai/") and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set in environment or .env file.\n"
                "Get your key from: https://platform.openai.com/api-keys"
            )

        if model.startswith("anthropic/") and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set in environment or .env file.\n"
                "Get your key from: https://console.anthropic.com/settings/keys"
            )

        # Ollama models run locally - no API key needed
        if model.startswith("ollama/"):
            pass
