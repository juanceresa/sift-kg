"""Tests for sift_kg.domains (models, loader)."""

from pathlib import Path

import pytest
import yaml

from sift_kg.domains.loader import DomainLoader, load_domain
from sift_kg.domains.models import DomainConfig, EntityTypeConfig, RelationTypeConfig


class TestDomainModels:
    """Test domain Pydantic models."""

    def test_entity_type_defaults(self):
        """EntityTypeConfig has sensible defaults."""
        et = EntityTypeConfig(description="A person")
        assert et.description == "A person"
        assert et.extraction_hints == []

    def test_entity_type_with_hints(self):
        """EntityTypeConfig accepts extraction hints."""
        et = EntityTypeConfig(description="An org", extraction_hints=["company", "firm"])
        assert "company" in et.extraction_hints

    def test_relation_type_defaults(self):
        """RelationTypeConfig has sensible defaults."""
        rt = RelationTypeConfig(description="Employment")
        assert rt.review_required is False
        assert rt.source_types == []
        assert rt.target_types == []

    def test_relation_type_review_required(self):
        """RelationTypeConfig can flag review_required."""
        rt = RelationTypeConfig(description="Risky", review_required=True)
        assert rt.review_required is True

    def test_domain_config_names(self, sample_domain):
        """DomainConfig exposes entity and relation type names."""
        entity_names = sample_domain.get_entity_type_names()
        assert "PERSON" in entity_names
        assert "ORGANIZATION" in entity_names

        relation_names = sample_domain.get_relation_type_names()
        assert "WORKS_FOR" in relation_names

    def test_domain_config_system_context(self, sample_domain):
        """DomainConfig stores system context."""
        assert sample_domain.system_context == "Test context for unit tests."

    def test_domain_config_no_context(self):
        """DomainConfig works without system context."""
        dc = DomainConfig(
            name="Minimal",
            entity_types={"THING": EntityTypeConfig(description="A thing")},
            relation_types={},
        )
        assert dc.system_context is None


class TestDomainLoader:
    """Test loading domains from YAML files."""

    def test_load_bundled_default(self):
        """Loading bundled 'default' domain works."""
        loader = DomainLoader()
        domain = loader.load_bundled("default")
        assert domain.name
        assert len(domain.entity_types) > 0
        assert len(domain.relation_types) > 0

    def test_load_from_custom_yaml(self, tmp_dir):
        """Loading from a custom YAML file works."""
        yaml_content = {
            "name": "Custom Domain",
            "entity_types": {
                "ANIMAL": {"description": "An animal"},
                "HABITAT": {"description": "Where animals live"},
            },
            "relation_types": {
                "LIVES_IN": {
                    "description": "Animal lives in habitat",
                    "source_types": ["ANIMAL"],
                    "target_types": ["HABITAT"],
                },
            },
        }
        yaml_path = tmp_dir / "custom.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        loader = DomainLoader()
        domain = loader.load_from_path(yaml_path)
        assert domain.name == "Custom Domain"
        assert "ANIMAL" in domain.entity_types
        assert "LIVES_IN" in domain.relation_types

    def test_load_from_nonexistent_path(self):
        """Loading from missing path raises error."""
        loader = DomainLoader()
        with pytest.raises((FileNotFoundError, Exception)):
            loader.load_from_path(Path("/nonexistent/domain.yaml"))

    def test_load_domain_convenience(self):
        """Module-level load_domain() loads default."""
        domain = load_domain()
        assert isinstance(domain, DomainConfig)
        assert domain.name

    def test_load_domain_custom_path(self, tmp_dir):
        """Module-level load_domain() with custom path."""
        yaml_content = {
            "name": "Via Convenience",
            "entity_types": {"X": {"description": "X"}},
            "relation_types": {},
        }
        yaml_path = tmp_dir / "domain.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        domain = load_domain(domain_path=yaml_path)
        assert domain.name == "Via Convenience"

    def test_simple_entity_format(self, tmp_dir):
        """YAML with simple string values for entity types."""
        yaml_content = {
            "name": "Simple",
            "entity_types": {
                "PERSON": "A human being",
                "ORG": "An organization",
            },
            "relation_types": {},
        }
        yaml_path = tmp_dir / "simple.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        loader = DomainLoader()
        domain = loader.load_from_path(yaml_path)
        assert "PERSON" in domain.entity_types
        assert domain.entity_types["PERSON"].description == "A human being"
