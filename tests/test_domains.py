"""Tests for sift_kg.domains (models, loader, discovery)."""

from pathlib import Path

import pytest
import yaml

from sift_kg.domains.discovery import (
    build_discovery_prompt,
    load_discovered_domain,
    save_discovered_domain,
)
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

    def test_load_bundled_schema_free(self):
        """Loading bundled 'schema-free' domain works."""
        loader = DomainLoader()
        domain = loader.load_bundled("schema-free")
        assert domain.name
        assert domain.schema_free is True

    def test_load_bundled_general(self):
        """Loading bundled 'general' domain works."""
        loader = DomainLoader()
        domain = loader.load_bundled("general")
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


class TestDiscovery:
    """Test LLM-driven schema discovery utilities."""

    def test_build_discovery_prompt_contains_samples(self):
        """Prompt includes all provided text samples."""
        samples = ["Sample about cats and dogs.", "Sample about finance and banking."]
        prompt = build_discovery_prompt(samples)
        assert "SAMPLE 1" in prompt
        assert "SAMPLE 2" in prompt
        assert "cats and dogs" in prompt
        assert "finance and banking" in prompt

    def test_build_discovery_prompt_with_system_context(self):
        """System context is injected into the prompt."""
        prompt = build_discovery_prompt(
            ["Some text."],
            system_context="Focus on biomedical entities.",
        )
        assert "Focus on biomedical entities" in prompt

    def test_build_discovery_prompt_truncates_long_samples(self):
        """Samples longer than 3000 chars are truncated."""
        long_sample = "x" * 5000
        prompt = build_discovery_prompt([long_sample])
        # The prompt should contain at most 3000 x's from the sample
        assert prompt.count("x") <= 3000

    def test_save_and_load_roundtrip(self, tmp_dir):
        """Saving a DomainConfig and loading it back produces equivalent data."""
        domain = DomainConfig(
            name="Test Discovery",
            version="1.0.0",
            description="Test domain",
            entity_types={
                "PERSON": EntityTypeConfig(description="A person"),
                "COMPANY": EntityTypeConfig(
                    description="A business",
                    extraction_hints=["corporation", "firm"],
                ),
            },
            relation_types={
                "WORKS_FOR": RelationTypeConfig(
                    description="Employment",
                    source_types=["PERSON"],
                    target_types=["COMPANY"],
                ),
            },
            schema_free=False,
        )

        path = tmp_dir / "discovered_domain.yaml"
        save_discovered_domain(domain, path)
        assert path.exists()

        loaded = load_discovered_domain(path)
        assert loaded is not None
        assert loaded.name == "Test Discovery"
        assert "PERSON" in loaded.entity_types
        assert "COMPANY" in loaded.entity_types
        assert loaded.entity_types["COMPANY"].extraction_hints == ["corporation", "firm"]
        assert "WORKS_FOR" in loaded.relation_types
        assert loaded.relation_types["WORKS_FOR"].source_types == ["PERSON"]

    def test_load_missing_returns_none(self, tmp_dir):
        """Loading from a nonexistent path returns None."""
        result = load_discovered_domain(tmp_dir / "nonexistent.yaml")
        assert result is None

    def test_load_corrupt_returns_none(self, tmp_dir):
        """Loading from a corrupt YAML returns None."""
        path = tmp_dir / "bad.yaml"
        path.write_text("{{{{invalid yaml content")
        result = load_discovered_domain(path)
        assert result is None

    def test_save_creates_parent_dirs(self, tmp_dir):
        """save_discovered_domain creates parent directories."""
        domain = DomainConfig(
            name="Nested",
            entity_types={"X": EntityTypeConfig(description="X")},
            relation_types={},
        )
        path = tmp_dir / "nested" / "deep" / "domain.yaml"
        save_discovered_domain(domain, path)
        assert path.exists()

    def test_roundtrip_preserves_system_context(self, tmp_dir):
        """System context survives save/load cycle."""
        domain = DomainConfig(
            name="With Context",
            system_context="Analyze government documents.",
            entity_types={"AGENCY": EntityTypeConfig(description="Govt agency")},
            relation_types={},
        )
        path = tmp_dir / "ctx_domain.yaml"
        save_discovered_domain(domain, path)

        loaded = load_discovered_domain(path)
        assert loaded is not None
        assert loaded.system_context == "Analyze government documents."
