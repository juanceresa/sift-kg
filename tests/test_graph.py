"""Tests for sift_kg.graph (KnowledgeGraph, builder, postprocessor)."""



from sift_kg.graph.builder import (
    _make_entity_id,
    _make_relation_id,
    build_graph,
    flag_relations_for_review,
)
from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.graph.postprocessor import remove_redundant_edges


class TestKnowledgeGraph:
    """Test KnowledgeGraph CRUD and serialization."""

    def test_add_entity(self):
        """Adding an entity creates a node."""
        kg = KnowledgeGraph()
        kg.add_entity("person:alice", "PERSON", "Alice", confidence=0.9)
        assert kg.entity_count == 1
        entity = kg.get_entity("person:alice")
        assert entity["name"] == "Alice"
        assert entity["confidence"] == 0.9

    def test_add_entity_merge_confidence(self):
        """Adding same entity twice keeps higher confidence."""
        kg = KnowledgeGraph()
        kg.add_entity("person:alice", "PERSON", "Alice", confidence=0.7)
        kg.add_entity("person:alice", "PERSON", "Alice", confidence=0.9)
        entity = kg.get_entity("person:alice")
        assert entity["confidence"] == 0.9

    def test_add_entity_merge_lower_confidence_ignored(self):
        """Adding same entity with lower confidence doesn't downgrade."""
        kg = KnowledgeGraph()
        kg.add_entity("person:alice", "PERSON", "Alice", confidence=0.9)
        kg.add_entity("person:alice", "PERSON", "Alice", confidence=0.5)
        entity = kg.get_entity("person:alice")
        assert entity["confidence"] == 0.9

    def test_add_entity_merge_source_documents(self):
        """Adding same entity from different docs merges source_documents."""
        kg = KnowledgeGraph()
        kg.add_entity("person:alice", "PERSON", "Alice", source_documents=["doc1"])
        kg.add_entity("person:alice", "PERSON", "Alice", source_documents=["doc2"])
        entity = kg.get_entity("person:alice")
        assert "doc1" in entity["source_documents"]
        assert "doc2" in entity["source_documents"]

    def test_add_entity_merge_no_duplicate_docs(self):
        """Merging same entity from same doc doesn't duplicate."""
        kg = KnowledgeGraph()
        kg.add_entity("person:alice", "PERSON", "Alice", source_documents=["doc1"])
        kg.add_entity("person:alice", "PERSON", "Alice", source_documents=["doc1"])
        entity = kg.get_entity("person:alice")
        assert entity["source_documents"].count("doc1") == 1

    def test_add_entity_merge_attributes(self):
        """Merging entities merges their attributes."""
        kg = KnowledgeGraph()
        kg.add_entity("person:alice", "PERSON", "Alice", attributes={"role": "CEO"})
        kg.add_entity("person:alice", "PERSON", "Alice", attributes={"age": "42"})
        entity = kg.get_entity("person:alice")
        assert entity["attributes"]["role"] == "CEO"
        assert entity["attributes"]["age"] == "42"

    def test_add_entity_merge_context(self):
        """Merging preserves context from first occurrence."""
        kg = KnowledgeGraph()
        kg.add_entity("person:alice", "PERSON", "Alice", context="First context")
        kg.add_entity("person:alice", "PERSON", "Alice", context="Second context")
        entity = kg.get_entity("person:alice")
        assert entity["context"] == "First context"

    def test_add_relation(self):
        """Adding a relation between existing entities works."""
        kg = KnowledgeGraph()
        kg.add_entity("person:alice", "PERSON", "Alice")
        kg.add_entity("org:acme", "ORGANIZATION", "Acme")
        result = kg.add_relation("r1", "person:alice", "org:acme", "WORKS_FOR")
        assert result is True
        assert kg.relation_count == 1

    def test_add_relation_missing_source(self):
        """Adding relation with missing source returns False."""
        kg = KnowledgeGraph()
        kg.add_entity("org:acme", "ORGANIZATION", "Acme")
        result = kg.add_relation("r1", "person:alice", "org:acme", "WORKS_FOR")
        assert result is False

    def test_add_relation_missing_target(self):
        """Adding relation with missing target returns False."""
        kg = KnowledgeGraph()
        kg.add_entity("person:alice", "PERSON", "Alice")
        result = kg.add_relation("r1", "person:alice", "org:acme", "WORKS_FOR")
        assert result is False

    def test_get_entity_missing(self):
        """Getting nonexistent entity returns None."""
        kg = KnowledgeGraph()
        assert kg.get_entity("nonexistent") is None

    def test_get_relations(self):
        """Get all relations for an entity."""
        kg = KnowledgeGraph()
        kg.add_entity("a", "PERSON", "A")
        kg.add_entity("b", "ORG", "B")
        kg.add_entity("c", "LOC", "C")
        kg.add_relation("r1", "a", "b", "WORKS_FOR")
        kg.add_relation("r2", "c", "a", "LOCATED_IN")

        out_rels = kg.get_relations("a", direction="out")
        assert len(out_rels) == 1

        in_rels = kg.get_relations("a", direction="in")
        assert len(in_rels) == 1

        both_rels = kg.get_relations("a", direction="both")
        assert len(both_rels) == 2

    def test_get_relations_missing_entity(self):
        """Get relations for nonexistent entity returns empty list."""
        kg = KnowledgeGraph()
        assert kg.get_relations("nonexistent") == []

    def test_export_roundtrip(self, tmp_dir):
        """Export and reload graph preserves structure."""
        kg = KnowledgeGraph()
        kg.add_entity("person:alice", "PERSON", "Alice", confidence=0.9)
        kg.add_entity("org:acme", "ORGANIZATION", "Acme")
        kg.add_relation("r1", "person:alice", "org:acme", "WORKS_FOR", confidence=0.8)

        path = tmp_dir / "graph.json"
        kg.save(path)
        assert path.exists()

        loaded = KnowledgeGraph.load(path)
        assert loaded.entity_count == 2
        assert loaded.relation_count == 1
        entity = loaded.get_entity("person:alice")
        assert entity["name"] == "Alice"
        assert entity["confidence"] == 0.9

    def test_export_metadata(self):
        """Export includes metadata with counts."""
        kg = KnowledgeGraph()
        kg.add_entity("person:alice", "PERSON", "Alice")
        data = kg.export()
        assert data["metadata"]["entity_count"] == 1
        assert "created_at" in data["metadata"]


class TestEntityIdGeneration:
    """Test entity ID normalization."""

    def test_basic_id(self):
        """Simple name generates clean ID."""
        assert _make_entity_id("Alice Smith", "PERSON") == "person:alice_smith"

    def test_unicode_normalization(self):
        """Accented characters are normalized."""
        eid = _make_entity_id("José García", "PERSON")
        assert "jose" in eid
        assert "garcia" in eid

    def test_special_characters_stripped(self):
        """Special characters replaced with underscores."""
        eid = _make_entity_id("Acme Corp.", "ORGANIZATION")
        assert "." not in eid
        assert "acme" in eid

    def test_no_double_underscores(self):
        """Multiple underscores are collapsed."""
        eid = _make_entity_id("A  &  B  Corp", "ORGANIZATION")
        assert "__" not in eid

    def test_case_insensitive(self):
        """Entity IDs are case-insensitive."""
        assert _make_entity_id("Alice", "PERSON") == _make_entity_id("alice", "PERSON")
        assert _make_entity_id("ALICE", "PERSON") == _make_entity_id("alice", "PERSON")

    def test_whitespace_handling(self):
        """Leading/trailing whitespace is stripped."""
        assert _make_entity_id("  Alice  ", "PERSON") == _make_entity_id("Alice", "PERSON")


class TestRelationIdGeneration:
    """Test relation ID generation."""

    def test_basic_relation_id(self):
        """Relation ID contains source, type, target, doc."""
        rid = _make_relation_id("person:alice", "org:acme", "WORKS_FOR", "doc1")
        assert "person:alice" in rid
        assert "org:acme" in rid
        assert "WORKS_FOR" in rid
        assert "doc1" in rid


class TestBuildGraph:
    """Test graph building from extractions."""

    def test_builds_from_extraction(self, sample_extraction):
        """Build graph from a single extraction."""
        kg = build_graph([sample_extraction], postprocess=False)
        # Should have entities + DOCUMENT node
        assert kg.entity_count >= 3  # Alice, Acme, New York + doc node

    def test_builds_relations(self, sample_extraction):
        """Relations are added between entities."""
        kg = build_graph([sample_extraction], postprocess=False)
        assert kg.relation_count > 0

    def test_skips_errored_extractions(self, sample_entities):
        """Extractions with errors are skipped."""
        from sift_kg.extract.models import DocumentExtraction

        errored = DocumentExtraction(
            document_id="bad_doc",
            document_path="/tmp/bad.txt",
            error="Processing failed",
        )
        good = DocumentExtraction(
            document_id="good_doc",
            document_path="/tmp/good.txt",
            entities=sample_entities,
            relations=[],
        )
        kg = build_graph([errored, good], postprocess=False)
        # Only good doc's entities should be present
        assert kg.entity_count >= len(sample_entities)

    def test_document_entity_added(self, sample_extraction):
        """DOCUMENT entity is added for each extraction."""
        kg = build_graph([sample_extraction], postprocess=False)
        doc_entity = kg.get_entity(f"doc:{sample_extraction.document_id}")
        assert doc_entity is not None
        assert doc_entity["entity_type"] == "DOCUMENT"

    def test_empty_extractions(self):
        """Building from empty list produces empty graph."""
        kg = build_graph([])
        assert kg.entity_count == 0
        assert kg.relation_count == 0


class TestFlagRelationsForReview:
    """Test relation flagging."""

    def test_flags_low_confidence(self, sample_graph):
        """Relations below threshold are flagged."""
        flagged = flag_relations_for_review(sample_graph, confidence_threshold=0.95)
        # Most relations have confidence < 0.95
        assert len(flagged) > 0

    def test_no_flags_above_threshold(self, sample_graph):
        """No flags when threshold is low enough."""
        flagged = flag_relations_for_review(sample_graph, confidence_threshold=0.0)
        assert len(flagged) == 0

    def test_flags_review_required_type(self, sample_graph):
        """Relations of types in review_types are always flagged."""
        # Add a relation of a review-required type
        sample_graph.add_entity("x", "PERSON", "X")
        sample_graph.add_entity("y", "PERSON", "Y")
        sample_graph.add_relation("r_review", "x", "y", "SUSPICIOUS", confidence=1.0)

        flagged = flag_relations_for_review(
            sample_graph,
            confidence_threshold=0.0,
            review_types={"SUSPICIOUS"},
        )
        suspicious = [f for f in flagged if f["relation_type"] == "SUSPICIOUS"]
        assert len(suspicious) == 1

    def test_skips_mentioned_in(self, sample_graph):
        """MENTIONED_IN relations are never flagged."""
        flagged = flag_relations_for_review(sample_graph, confidence_threshold=0.0)
        mentioned = [f for f in flagged if f["relation_type"] == "MENTIONED_IN"]
        assert len(mentioned) == 0


class TestPostprocessor:
    """Test graph post-processing."""

    def test_remove_self_loops(self):
        """Self-loops are removed."""
        kg = KnowledgeGraph()
        kg.add_entity("a", "PERSON", "A")
        kg.graph.add_edge("a", "a", key="loop", relation_type="KNOWS")

        stats = remove_redundant_edges(kg)
        assert stats["self_loops_removed"] == 1
        assert kg.relation_count == 0

    def test_remove_transitive_located_in(self):
        """Transitive LOCATED_IN edges are removed."""
        kg = KnowledgeGraph()
        kg.add_entity("a", "ORG", "A")
        kg.add_entity("b", "LOC", "B")
        kg.add_entity("c", "LOC", "C")

        # A in B, B in C, A in C (redundant)
        kg.add_relation("r1", "a", "b", "LOCATED_IN")
        kg.add_relation("r2", "b", "c", "LOCATED_IN")
        kg.add_relation("r3", "a", "c", "LOCATED_IN")  # Should be removed

        assert kg.relation_count == 3
        stats = remove_redundant_edges(kg)
        assert stats["transitive_removed"] == 1
        assert kg.relation_count == 2

    def test_dry_run_no_modification(self):
        """Dry run reports stats but doesn't modify graph."""
        kg = KnowledgeGraph()
        kg.add_entity("a", "PERSON", "A")
        kg.graph.add_edge("a", "a", key="loop", relation_type="KNOWS")

        stats = remove_redundant_edges(kg, dry_run=True)
        assert stats["self_loops_removed"] == 1
        assert kg.relation_count == 1  # Still there

    def test_non_transitive_relations_untouched(self):
        """Non-transitive relation types are not removed."""
        kg = KnowledgeGraph()
        kg.add_entity("a", "PERSON", "A")
        kg.add_entity("b", "ORG", "B")
        kg.add_entity("c", "ORG", "C")

        # A employed_by B, B employed_by C, A employed_by C — NOT redundant
        kg.add_relation("r1", "a", "b", "EMPLOYED_BY")
        kg.add_relation("r2", "b", "c", "EMPLOYED_BY")
        kg.add_relation("r3", "a", "c", "EMPLOYED_BY")

        stats = remove_redundant_edges(kg)
        assert stats["transitive_removed"] == 0
        assert kg.relation_count == 3
