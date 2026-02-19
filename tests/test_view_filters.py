"""Tests for sift view pre-filters."""

import pytest

from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.visualize import filter_graph


def _make_test_graph() -> KnowledgeGraph:
    """Build a small graph for filter tests.

    Topology:
        alice --WORKS_FOR--> acme (conf 0.9, doc1)
        bob --WORKS_FOR--> acme (conf 0.4, doc2)
        carol --KNOWS--> alice (conf 0.8, doc1)
        carol --LOCATED_IN--> nyc (conf 0.7, doc2)
        dave --KNOWS--> bob (conf 0.3, doc3)
    Degrees: acme=2, alice=2, carol=2, bob=2, nyc=1, dave=1
    """
    kg = KnowledgeGraph()
    kg.add_entity("person:alice", "PERSON", "Alice", confidence=0.9, source_documents=["doc1"])
    kg.add_entity("person:bob", "PERSON", "Bob", confidence=0.4, source_documents=["doc2"])
    kg.add_entity("person:carol", "PERSON", "Carol", confidence=0.8, source_documents=["doc1", "doc2"])
    kg.add_entity("person:dave", "PERSON", "Dave", confidence=0.3, source_documents=["doc3"])
    kg.add_entity("org:acme", "ORGANIZATION", "Acme", confidence=0.9, source_documents=["doc1", "doc2"])
    kg.add_entity("location:nyc", "LOCATION", "New York City", confidence=0.7, source_documents=["doc2"])

    kg.add_relation("r1", "person:alice", "org:acme", "WORKS_FOR", confidence=0.9, source_document="doc1")
    kg.add_relation("r2", "person:bob", "org:acme", "WORKS_FOR", confidence=0.4, source_document="doc2")
    kg.add_relation("r3", "person:carol", "person:alice", "KNOWS", confidence=0.8, source_document="doc1")
    kg.add_relation("r4", "person:carol", "location:nyc", "LOCATED_IN", confidence=0.7, source_document="doc2")
    kg.add_relation("r5", "person:dave", "person:bob", "KNOWS", confidence=0.3, source_document="doc3")
    return kg


class TestFilterGraph:
    """Test pre-filter logic."""

    def test_no_filters_returns_same_graph(self):
        """No filters preserves all nodes and edges."""
        kg = _make_test_graph()
        result = filter_graph(kg)
        assert result.entity_count == 6
        assert result.relation_count == 5

    def test_top_n(self):
        """Top N keeps highest-degree hubs plus their direct neighbors."""
        kg = _make_test_graph()
        # top_n=1: hub is org:acme (degree 2, first alphabetically among ties)
        # neighbors of acme: alice, bob → 3 nodes, with edges between them
        result = filter_graph(kg, top_n=1)
        node_ids = set(result.graph.nodes())
        assert "org:acme" in node_ids
        assert "person:alice" in node_ids
        assert "person:bob" in node_ids
        assert result.entity_count == 3
        assert result.relation_count >= 2

    def test_min_confidence(self):
        """Removes nodes and edges below confidence threshold."""
        kg = _make_test_graph()
        result = filter_graph(kg, min_confidence=0.7)
        assert result.entity_count == 4
        node_ids = list(result.graph.nodes())
        assert "person:bob" not in node_ids
        assert "person:dave" not in node_ids

    def test_source_doc(self):
        """Only entities/edges from the specified document are retained."""
        kg = _make_test_graph()
        result = filter_graph(kg, source_doc="doc1")
        # Entities with doc1: alice, carol, acme. Edges from doc1: r1, r3.
        assert result.entity_count == 3
        assert result.relation_count == 2

    def test_neighborhood_depth_1(self):
        """1-hop neighborhood includes direct neighbors only."""
        kg = _make_test_graph()
        result = filter_graph(kg, neighborhood="person:alice", depth=1)
        node_ids = list(result.graph.nodes())
        assert "person:alice" in node_ids
        assert "org:acme" in node_ids
        assert "person:carol" in node_ids
        assert "person:dave" not in node_ids

    def test_neighborhood_depth_2(self):
        """2-hop neighborhood includes neighbors-of-neighbors."""
        kg = _make_test_graph()
        result = filter_graph(kg, neighborhood="person:alice", depth=2)
        node_ids = list(result.graph.nodes())
        assert "person:bob" in node_ids
        assert "location:nyc" in node_ids
        # dave is 3 hops away (alice->acme->bob->dave)
        assert "person:dave" not in node_ids

    def test_neighborhood_invalid_entity(self):
        """Neighborhood with nonexistent entity raises ValueError."""
        kg = _make_test_graph()
        with pytest.raises(ValueError, match="not found"):
            filter_graph(kg, neighborhood="person:nonexistent", depth=1)

    def test_combined_filters(self):
        """Multiple filters compose: min_confidence then top_n (hubs + neighbors)."""
        kg = _make_test_graph()
        # min_confidence=0.7 drops bob, dave → 4 left (alice, carol, acme, nyc)
        # top_n=2 picks top 2 hubs (alice, carol both degree 2) + their neighbors → all 4
        result = filter_graph(kg, min_confidence=0.7, top_n=2)
        assert result.entity_count == 4
        assert result.relation_count >= 2
