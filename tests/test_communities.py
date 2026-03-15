"""Tests for sift_kg.graph.communities."""

import pytest

from sift_kg.graph.knowledge_graph import KnowledgeGraph


def _build_two_cluster_graph() -> KnowledgeGraph:
    """Build a graph with two clearly separable clusters.

    Cluster A: persons a1-a10 all connected to each other
    Cluster B: persons b1-b10 all connected to each other
    Single bridge: a1 -> b1
    """
    kg = KnowledgeGraph()
    # Cluster A
    for i in range(1, 11):
        kg.add_entity(f"person:a{i}", "PERSON", f"A{i}")
    for i in range(1, 11):
        for j in range(i + 1, 11):
            kg.add_relation(f"r_a{i}_{j}", f"person:a{i}", f"person:a{j}", "ASSOCIATED_WITH")

    # Cluster B
    for i in range(1, 11):
        kg.add_entity(f"person:b{i}", "PERSON", f"B{i}")
    for i in range(1, 11):
        for j in range(i + 1, 11):
            kg.add_relation(f"r_b{i}_{j}", f"person:b{i}", f"person:b{j}", "ASSOCIATED_WITH")

    # Bridge
    kg.add_relation("r_bridge", "person:a1", "person:b1", "ASSOCIATED_WITH")

    # Add DOCUMENT nodes and MENTIONED_IN edges (should be stripped)
    kg.add_entity("doc:test", "DOCUMENT", "test")
    for i in range(1, 11):
        kg.add_relation(f"r_doc_a{i}", f"person:a{i}", "doc:test", "MENTIONED_IN", canonicalize=False)

    return kg


class TestDetectCommunities:
    """Test community detection."""

    def test_detects_two_communities(self):
        """Two dense clusters should be detected as separate communities."""
        from sift_kg.graph.communities import detect_communities

        kg = _build_two_cluster_graph()
        communities = detect_communities(kg, min_community_size=3)
        assert communities is not None
        assert len(communities) >= 2

    def test_strips_document_nodes(self):
        """DOCUMENT nodes should not appear in community members."""
        from sift_kg.graph.communities import detect_communities

        kg = _build_two_cluster_graph()
        communities = detect_communities(kg, min_community_size=3)
        assert communities is not None
        all_ids = [e["id"] for comm in communities for e in comm]
        assert not any(eid.startswith("doc:") for eid in all_ids)

    def test_returns_none_for_tiny_graph(self):
        """Graph too small for meaningful communities returns None."""
        from sift_kg.graph.communities import detect_communities

        kg = KnowledgeGraph()
        kg.add_entity("person:a", "PERSON", "A")
        kg.add_entity("person:b", "PERSON", "B")
        kg.add_relation("r1", "person:a", "person:b", "KNOWS")
        result = detect_communities(kg)
        assert result is None

    def test_described_ids_filters_members(self):
        """Only entities in described_ids appear in community output."""
        from sift_kg.graph.communities import detect_communities

        kg = _build_two_cluster_graph()
        # Only include half the entities
        described = {f"person:a{i}" for i in range(1, 6)}
        communities = detect_communities(kg, described_ids=described, min_community_size=3)
        if communities:
            all_ids = {e["id"] for comm in communities for e in comm}
            assert all_ids.issubset(described)

    def test_sorted_by_total_degree(self):
        """Communities are sorted by total degree (highest first)."""
        from sift_kg.graph.communities import detect_communities

        kg = _build_two_cluster_graph()
        communities = detect_communities(kg, min_community_size=3)
        if communities and len(communities) >= 2:
            degree_map = dict(kg.graph.degree())
            deg_0 = sum(degree_map.get(e["id"], 0) for e in communities[0])
            deg_1 = sum(degree_map.get(e["id"], 0) for e in communities[1])
            assert deg_0 >= deg_1
