"""Domain configuration loader.

Loads and validates domain configurations from YAML files.
Supports both bundled domains (shipped with sift-kg) and
user-provided custom domain files.
"""

import logging
from pathlib import Path

import yaml

from sift_kg.domains.models import DomainConfig, EntityTypeConfig, RelationTypeConfig

logger = logging.getLogger(__name__)

# Bundled domains directory (shipped with the package)
BUNDLED_DOMAINS_DIR = Path(__file__).parent / "bundled"

# Module-level loader for convenience function
_default_loader: "DomainLoader | None" = None


def load_domain(
    domain_path: Path | None = None,
    bundled_name: str = "default",
) -> DomainConfig:
    """Convenience function to load a domain configuration.

    Args:
        domain_path: Path to custom domain YAML (takes priority)
        bundled_name: Name of bundled domain to load if no path given

    Returns:
        Validated DomainConfig
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = DomainLoader()
    if domain_path:
        return _default_loader.load_from_path(domain_path)
    return _default_loader.load_bundled(bundled_name)


class DomainLoader:
    """Load and validate domain configurations from YAML files."""

    def __init__(self) -> None:
        self._cache: dict[str, DomainConfig] = {}

    def load_from_path(self, yaml_path: Path) -> DomainConfig:
        """Load a domain configuration from a specific YAML file.

        Args:
            yaml_path: Path to domain.yaml file

        Returns:
            Validated DomainConfig

        Raises:
            ValueError: If file not found or invalid
        """
        yaml_path = Path(yaml_path)
        cache_key = str(yaml_path.resolve())

        if cache_key in self._cache:
            return self._cache[cache_key]

        if not yaml_path.exists():
            raise ValueError(f"Domain config not found: {yaml_path}")

        logger.info(f"Loading domain configuration: {yaml_path}")
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

        # Support both top-level and nested 'domain:' key
        if "domain" in raw:
            raw = raw["domain"]

        config = self._parse_config(raw)
        self._cache[cache_key] = config

        logger.info(
            f"Loaded domain '{config.name}' "
            f"({len(config.entity_types)} entity types, "
            f"{len(config.relation_types)} relation types)"
        )
        return config

    def load_bundled(self, name: str = "default") -> DomainConfig:
        """Load a bundled domain by name.

        Args:
            name: Bundled domain name (default: "default")

        Returns:
            Validated DomainConfig
        """
        domain_path = BUNDLED_DOMAINS_DIR / name / "domain.yaml"
        if not domain_path.exists():
            available = self.list_bundled()
            raise ValueError(
                f"Bundled domain '{name}' not found. Available: {available}"
            )
        return self.load_from_path(domain_path)

    def list_bundled(self) -> list[str]:
        """List available bundled domain names."""
        if not BUNDLED_DOMAINS_DIR.exists():
            return []
        return sorted(
            d.name
            for d in BUNDLED_DOMAINS_DIR.iterdir()
            if d.is_dir() and (d / "domain.yaml").exists()
        )

    def _parse_config(self, raw: dict) -> DomainConfig:
        """Parse raw YAML dict into DomainConfig."""
        entity_types = {}
        for name, cfg in raw.get("entity_types", {}).items():
            if isinstance(cfg, str):
                # Simple form: "PERSON: People and individuals"
                entity_types[name] = EntityTypeConfig(description=cfg)
            else:
                entity_types[name] = EntityTypeConfig(
                    description=cfg.get("description", ""),
                    extraction_hints=cfg.get("extraction_hints", []),
                    canonical_names=cfg.get("canonical_names", []),
                    canonical_fallback_type=cfg.get("canonical_fallback_type"),
                )

        relation_types = {}
        for name, cfg in raw.get("relation_types", {}).items():
            if isinstance(cfg, str):
                relation_types[name] = RelationTypeConfig(description=cfg)
            else:
                relation_types[name] = RelationTypeConfig(
                    description=cfg.get("description", ""),
                    source_types=cfg.get("source_types", []),
                    target_types=cfg.get("target_types", []),
                    symmetric=cfg.get("symmetric", False),
                    extraction_hints=cfg.get("extraction_hints", []),
                    review_required=cfg.get("review_required", False),
                )

        return DomainConfig(
            name=raw.get("name", "Unknown"),
            version=raw.get("version", "1.0.0"),
            description=raw.get("description", ""),
            entity_types=entity_types,
            relation_types=relation_types,
            system_context=raw.get("system_context"),
        )
