"""Tests for sift_kg.narrate.prompts."""


from sift_kg.narrate.prompts import build_entity_description_prompt, build_narrative_prompt


class TestNarrativePrompt:
    """Test narrative overview prompt generation."""

    def test_includes_entity_names(self):
        """Prompt includes entity names."""
        entities = [
            {"name": "Alice", "entity_type": "PERSON"},
            {"name": "Acme Corp", "entity_type": "ORGANIZATION"},
        ]
        prompt = build_narrative_prompt(entities, [], document_count=1)
        assert "Alice" in prompt
        assert "Acme Corp" in prompt

    def test_includes_entity_type_counts(self):
        """Prompt groups entities by type with counts."""
        entities = [
            {"name": "Alice", "entity_type": "PERSON"},
            {"name": "Bob", "entity_type": "PERSON"},
            {"name": "Acme", "entity_type": "ORGANIZATION"},
        ]
        prompt = build_narrative_prompt(entities, [], document_count=1)
        assert "PERSON" in prompt
        assert "ORGANIZATION" in prompt

    def test_includes_relations(self):
        """Prompt includes relation information."""
        entities = [{"name": "Alice", "entity_type": "PERSON"}]
        relations = [
            {
                "source_name": "Alice",
                "target_name": "Acme",
                "relation_type": "WORKS_FOR",
            }
        ]
        prompt = build_narrative_prompt(entities, relations, document_count=1)
        assert "Alice" in prompt
        assert "WORKS_FOR" in prompt

    def test_includes_document_count(self):
        """Prompt mentions document count."""
        entities = [{"name": "X", "entity_type": "THING"}]
        prompt = build_narrative_prompt(entities, [], document_count=42)
        assert "42" in prompt

    def test_includes_system_context(self):
        """System context is included when provided."""
        entities = [{"name": "X", "entity_type": "THING"}]
        prompt = build_narrative_prompt(
            entities, [], document_count=1, system_context="Cuban land rights domain"
        )
        assert "Cuban land rights" in prompt

    def test_includes_all_relations_passed(self):
        """Prompt includes all relations it receives (caller pre-filters)."""
        entities = [{"name": "X", "entity_type": "THING"}]
        relations = [
            {"source_name": f"A{i}", "target_name": f"B{i}", "relation_type": "RELATES"}
            for i in range(20)
        ]
        prompt = build_narrative_prompt(entities, relations, document_count=1)
        assert "A19" in prompt

    def test_scope_note_when_truncated(self):
        """Prompt shows scope note when total counts exceed what's passed."""
        entities = [
            {"name": f"Person{i}", "entity_type": "PERSON"} for i in range(10)
        ]
        prompt = build_narrative_prompt(
            entities, [], document_count=1,
            total_entities=200, total_relations=500,
        )
        assert "most connected" in prompt.lower()
        assert "200" in prompt


class TestEntityDescriptionPrompt:
    """Test entity description prompt generation."""

    def test_includes_entity_name(self):
        """Prompt includes the entity name."""
        prompt = build_entity_description_prompt(
            entity_name="Alice Smith",
            entity_type="PERSON",
            attributes={},
            relations=[],
            source_documents=[],
        )
        assert "Alice Smith" in prompt

    def test_includes_entity_type(self):
        """Prompt includes the entity type."""
        prompt = build_entity_description_prompt(
            entity_name="Alice",
            entity_type="PERSON",
            attributes={},
            relations=[],
            source_documents=[],
        )
        assert "PERSON" in prompt

    def test_includes_attributes(self):
        """Prompt includes entity attributes."""
        prompt = build_entity_description_prompt(
            entity_name="Alice",
            entity_type="PERSON",
            attributes={"role": "CEO", "age": "42"},
            relations=[],
            source_documents=[],
        )
        assert "CEO" in prompt

    def test_includes_relations(self):
        """Prompt includes entity relations."""
        relations = [
            {"source_name": "Alice", "target_name": "Acme", "relation_type": "WORKS_FOR"},
        ]
        prompt = build_entity_description_prompt(
            entity_name="Alice",
            entity_type="PERSON",
            attributes={},
            relations=relations,
            source_documents=[],
        )
        assert "WORKS_FOR" in prompt
        assert "Acme" in prompt

    def test_excludes_source_document_filenames(self):
        """Prompt no longer includes raw document filenames (noisy)."""
        prompt = build_entity_description_prompt(
            entity_name="Alice",
            entity_type="PERSON",
            attributes={},
            relations=[],
            source_documents=["report_2024.pdf", "interview.txt"],
        )
        assert "report_2024.pdf" not in prompt

    def test_no_relations_placeholder(self):
        """Prompt handles entity with no relations."""
        prompt = build_entity_description_prompt(
            entity_name="Unknown",
            entity_type="PERSON",
            attributes={},
            relations=[],
            source_documents=[],
        )
        assert "No known relations" in prompt or "Unknown" in prompt
