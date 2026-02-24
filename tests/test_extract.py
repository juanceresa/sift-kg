"""Tests for sift_kg.extract.models and sift_kg.extract.prompts."""

from sift_kg.extract.models import (
    DocumentExtraction,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from sift_kg.extract.prompts import build_combined_prompt


class TestExtractModels:
    """Test extraction Pydantic models."""

    def test_entity_defaults(self):
        """ExtractedEntity has sensible defaults."""
        e = ExtractedEntity(name="Test", entity_type="PERSON")
        assert e.confidence == 0.5
        assert e.context == ""
        assert e.attributes == {}

    def test_entity_with_all_fields(self):
        """ExtractedEntity with all fields set."""
        e = ExtractedEntity(
            name="Alice",
            entity_type="PERSON",
            confidence=0.9,
            context="Alice is the CEO.",
            attributes={"role": "CEO"},
        )
        assert e.name == "Alice"
        assert e.attributes["role"] == "CEO"

    def test_relation_defaults(self):
        """ExtractedRelation has sensible defaults."""
        r = ExtractedRelation(
            source_entity="Alice",
            target_entity="Acme",
            relation_type="WORKS_FOR",
        )
        assert r.confidence == 0.5
        assert r.evidence == ""

    def test_extraction_result(self):
        """ExtractionResult holds entities and relations from a chunk."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="A", entity_type="PERSON"),
            ],
            relations=[
                ExtractedRelation(
                    source_entity="A",
                    target_entity="B",
                    relation_type="KNOWS",
                ),
            ],
        )
        assert len(result.entities) == 1
        assert len(result.relations) == 1

    def test_document_extraction_defaults(self):
        """DocumentExtraction has sensible defaults."""
        de = DocumentExtraction(document_id="doc1", document_path="/tmp/doc1.txt")
        assert de.entities == []
        assert de.relations == []
        assert de.error is None
        assert de.cost_usd == 0.0

    def test_document_extraction_with_error(self):
        """DocumentExtraction can carry an error message."""
        de = DocumentExtraction(
            document_id="doc1",
            document_path="/tmp/doc1.txt",
            error="Processing failed",
        )
        assert de.error == "Processing failed"


class TestExtractPrompts:
    """Test prompt generation."""

    def test_combined_prompt_contains_entity_types(self, sample_domain):
        """Combined prompt includes entity type names."""
        prompt = build_combined_prompt("Some text about people.", "test_doc", sample_domain)
        assert "PERSON" in prompt
        assert "ORGANIZATION" in prompt
        assert "LOCATION" in prompt

    def test_combined_prompt_contains_text(self, sample_domain):
        """Combined prompt includes the input text."""
        text = "Alice works at Acme Corp."
        prompt = build_combined_prompt(text, "test_doc", sample_domain)
        assert text in prompt

    def test_combined_prompt_contains_hints(self, sample_domain):
        """Combined prompt includes extraction hints."""
        prompt = build_combined_prompt("text", "test_doc", sample_domain)
        assert "company" in prompt or "agency" in prompt

    def test_combined_prompt_contains_relation_types(self, sample_domain):
        """Combined prompt includes relation type names."""
        prompt = build_combined_prompt("text", "test_doc", sample_domain)
        assert "WORKS_FOR" in prompt
        assert "LOCATED_IN" in prompt

    def test_combined_prompt_has_json_schema(self, sample_domain):
        """Combined prompt contains JSON output format."""
        prompt = build_combined_prompt("text", "test_doc", sample_domain)
        assert "JSON" in prompt or "json" in prompt

    def test_combined_prompt_includes_context(self, sample_domain):
        """Combined prompt includes domain system context."""
        prompt = build_combined_prompt("text", "test_doc", sample_domain)
        assert "Test context" in prompt

    def test_combined_prompt_enforces_entity_types(self, sample_domain):
        """Combined prompt tells LLM to use only defined entity types."""
        prompt = build_combined_prompt("text", "test_doc", sample_domain)
        assert "use ONLY these" in prompt
        assert "Do not create new entity types" in prompt

    def test_combined_prompt_enforces_relation_types(self, sample_domain):
        """Combined prompt tells LLM to use only defined relation types."""
        prompt = build_combined_prompt("text", "test_doc", sample_domain)
        assert "do not invent new types" in prompt

    def test_combined_prompt_fallback_when_set(self, sample_domain):
        """Combined prompt uses fallback_relation when defined."""
        sample_domain.fallback_relation = "ASSOCIATED_WITH"
        prompt = build_combined_prompt("text", "test_doc", sample_domain)
        assert "use ASSOCIATED_WITH" in prompt

    def test_combined_prompt_strict_when_no_fallback(self, sample_domain):
        """Combined prompt enforces strict matching when no fallback."""
        sample_domain.fallback_relation = None
        prompt = build_combined_prompt("text", "test_doc", sample_domain)
        assert "Only extract relationships that clearly match" in prompt

    def test_combined_prompt_direction_hints(self, sample_domain):
        """Combined prompt includes relation direction hints."""
        prompt = build_combined_prompt("text", "test_doc", sample_domain)
        assert "PERSON" in prompt
        assert "ORGANIZATION" in prompt
        # WORKS_FOR has source/target constraints
        assert "→" in prompt
