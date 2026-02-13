"""Configuration management for sift-kg using pydantic-settings.

This module provides the SiftConfig class for managing application settings
from environment variables and .env files. All configuration is type-safe
and validated using Pydantic models.

Settings priority (highest to lowest):
1. CLI flags (applied after SiftConfig creation)
2. Environment variables (SIFT_* prefix)
3. .env file
4. sift.yaml project config
5. Default values
"""

import logging
import os
from pathlib import Path

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

logger = logging.getLogger(__name__)

# Map sift.yaml keys to SiftConfig field names
_YAML_TO_FIELD = {
    "domain": "domain_path",
    "model": "default_model",
    "output": "output_dir",
}


class _ProjectYamlSource(PydanticBaseSettingsSource):
    """Read project config from sift.yaml (lower priority than env vars)."""

    def get_field_value(self, field, field_name):  # type: ignore[override]
        return None, field_name, False

    def __call__(self) -> dict:
        project_file = Path("sift.yaml")
        if not project_file.exists():
            return {}

        raw = yaml.safe_load(project_file.read_text()) or {}

        result: dict = {}
        for yaml_key, field_name in _YAML_TO_FIELD.items():
            if yaml_key in raw:
                result[field_name] = raw[yaml_key]
        return result


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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _ProjectYamlSource(settings_cls),
            file_secret_settings,
        )

    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for GPT models. Get from: https://platform.openai.com/api-keys"
    )

    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for Claude models. Get from: https://console.anthropic.com/settings/keys"
    )

    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key. Get from: https://aistudio.google.com/apikey"
    )

    default_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Default LLM model in format provider/model-name (e.g., openai/gpt-4o-mini, anthropic/claude-haiku, ollama/llama3.3)"
    )

    output_dir: Path = Field(
        default=Path("output"),
        description="Directory for output files (extractions, graph, narrative)"
    )

    domain_path: Path | None = Field(
        default=None,
        description="Path to custom domain YAML file (uses default domain if not set)"
    )

    @model_validator(mode="after")
    def _export_api_keys(self) -> "SiftConfig":
        """Export API keys to environment so LiteLLM can find them."""
        key_map = {
            "OPENAI_API_KEY": self.openai_api_key,
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
            "GEMINI_API_KEY": self.gemini_api_key,
        }
        for env_var, value in key_map.items():
            if value and env_var not in os.environ:
                os.environ[env_var] = value
        return self

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
        if model.startswith("openai/") and not self.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found. Set in environment or .env file.\n"
                "Get your key from: https://platform.openai.com/api-keys"
            )

        if model.startswith("anthropic/") and not self.anthropic_api_key and not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set in environment or .env file.\n"
                "Get your key from: https://console.anthropic.com/settings/keys"
            )

        if model.startswith("gemini/") and not self.gemini_api_key and not os.environ.get("GEMINI_API_KEY"):
            raise ValueError(
                "GEMINI_API_KEY not found. Set in environment or .env file.\n"
                "Get your key from: https://aistudio.google.com/apikey"
            )

        # Ollama models run locally - no API key needed
        if model.startswith("ollama/"):
            pass
