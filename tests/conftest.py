"""Shared test fixtures for sift-kg."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sift_kg.domains.models import DomainConfig, EntityTypeConfig, RelationTypeConfig
from sift_kg.extract.models import (
    DocumentExtraction,
    ExtractedEntity,
    ExtractedRelation,
)
from sift_kg.graph.knowledge_graph import KnowledgeGraph


@pytest.fixture
def sample_domain() -> DomainConfig:
    """Minimal domain config for testing."""
    return DomainConfig(
        name="Test Domain",
        entity_types={
            "PERSON": EntityTypeConfig(description="A person"),
            "ORGANIZATION": EntityTypeConfig(
                description="An org", extraction_hints=["company", "agency"]
            ),
            "LOCATION": EntityTypeConfig(description="A place"),
        },
        relation_types={
            "WORKS_FOR": RelationTypeConfig(
                description="Employment",
                source_types=["PERSON"],
                target_types=["ORGANIZATION"],
            ),
            "LOCATED_IN": RelationTypeConfig(
                description="Location containment",
                source_types=["ORGANIZATION", "PERSON"],
                target_types=["LOCATION"],
            ),
            "ASSOCIATED_WITH": RelationTypeConfig(
                description="General association",
                review_required=True,
            ),
        },
        system_context="Test context for unit tests.",
    )


@pytest.fixture
def sample_entities() -> list[ExtractedEntity]:
    """Sample extracted entities for testing."""
    return [
        ExtractedEntity(
            name="Alice Smith",
            entity_type="PERSON",
            confidence=0.9,
            context="Alice Smith is the CEO.",
            attributes={"role": "CEO"},
        ),
        ExtractedEntity(
            name="Acme Corp",
            entity_type="ORGANIZATION",
            confidence=0.85,
            context="Acme Corp is based in NYC.",
            attributes={"industry": "tech"},
        ),
        ExtractedEntity(
            name="New York",
            entity_type="LOCATION",
            confidence=0.95,
            context="Located in New York.",
            attributes={},
        ),
    ]


@pytest.fixture
def sample_relations() -> list[ExtractedRelation]:
    """Sample extracted relations for testing."""
    return [
        ExtractedRelation(
            source_entity="Alice Smith",
            target_entity="Acme Corp",
            relation_type="WORKS_FOR",
            confidence=0.8,
            evidence="Alice Smith is the CEO of Acme Corp.",
        ),
        ExtractedRelation(
            source_entity="Acme Corp",
            target_entity="New York",
            relation_type="LOCATED_IN",
            confidence=0.9,
            evidence="Acme Corp is based in New York.",
        ),
    ]


@pytest.fixture
def sample_extraction(
    sample_entities, sample_relations
) -> DocumentExtraction:
    """A complete DocumentExtraction for testing."""
    return DocumentExtraction(
        document_id="test_doc",
        document_path="/tmp/test_doc.txt",
        entities=sample_entities,
        relations=sample_relations,
        chunks_processed=3,
        model_used="test-model",
        cost_usd=0.001,
    )


@pytest.fixture
def sample_graph(sample_extraction) -> KnowledgeGraph:
    """Pre-built KnowledgeGraph for testing."""
    from sift_kg.graph.builder import build_graph

    return build_graph([sample_extraction], postprocess=False)


@pytest.fixture
def tmp_dir():
    """Temporary directory that cleans up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def mock_llm():
    """Mock LLMClient that returns configurable responses."""
    llm = MagicMock()
    llm.model = "test-model"
    llm.total_cost_usd = 0.0
    llm.total_input_tokens = 0
    llm.total_output_tokens = 0
    return llm
