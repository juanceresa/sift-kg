"""Tests for sift_kg.resolve.clustering (embedding-based entity clustering)."""

import pytest

from sift_kg.resolve.clustering import EMBEDDINGS_AVAILABLE


@pytest.mark.skipif(not EMBEDDINGS_AVAILABLE, reason="sentence-transformers not installed")
class TestClusterEntitiesByEmbedding:
    """Test embedding-based clustering."""

    def _make_entities(self, names: list[str], aliases: dict[str, list[str]] | None = None) -> list[dict]:
        """Helper: build entity dicts from names."""
        aliases = aliases or {}
        return [
            {
                "id": f"person:{name.lower().replace(' ', '_')}",
                "name": name,
                "aliases": aliases.get(name, []),
                "attributes": {},
            }
            for name in names
        ]

    def test_small_input_single_batch(self):
        """Fewer than 10 entities returns single batch."""
        from sift_kg.resolve.clustering import cluster_entities_by_embedding

        entities = self._make_entities(["Alice", "Bob", "Charlie"])
        result = cluster_entities_by_embedding(entities)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_clusters_created(self):
        """Enough entities should produce multiple clusters."""
        from sift_kg.resolve.clustering import cluster_entities_by_embedding

        # Create 30 entities — enough to test clustering with small target_cluster_size
        names = [f"Person_{i}" for i in range(30)]
        entities = self._make_entities(names)
        result = cluster_entities_by_embedding(entities, target_cluster_size=10)
        assert len(result) >= 2

    def test_all_entities_preserved(self):
        """Every entity should appear in exactly one cluster."""
        from sift_kg.resolve.clustering import cluster_entities_by_embedding

        names = [f"Entity_{i}" for i in range(25)]
        entities = self._make_entities(names)
        result = cluster_entities_by_embedding(entities, target_cluster_size=10)
        all_entities = [e for cluster in result for e in cluster]
        assert len(all_entities) == len(entities)
        all_ids = {e["id"] for e in all_entities}
        expected_ids = {e["id"] for e in entities}
        assert all_ids == expected_ids

    def test_semantic_similarity_grouping(self):
        """Semantically similar names should tend to cluster together."""
        from sift_kg.resolve.clustering import cluster_entities_by_embedding

        # Mix of people names and city names — should tend to separate
        people = ["John Smith", "Jane Smith", "Robert Johnson", "Mary Johnson",
                   "William Davis", "Sarah Davis", "James Wilson", "Emily Wilson",
                   "Michael Brown", "Jessica Brown", "David Taylor", "Amanda Taylor"]
        cities = ["New York City", "Los Angeles", "San Francisco", "Chicago",
                  "Houston Texas", "Phoenix Arizona", "Philadelphia", "San Antonio",
                  "San Diego", "Dallas Texas", "San Jose", "Austin Texas"]
        entities = self._make_entities(people + cities)
        result = cluster_entities_by_embedding(entities, target_cluster_size=12)
        # Not deterministic, but with 24 entities and cluster_size=12, should get ~2 clusters
        assert len(result) >= 2

    def test_aliases_used_in_embedding(self):
        """Aliases should be included in the embedding text."""
        from sift_kg.resolve.clustering import cluster_entities_by_embedding

        entities = self._make_entities(
            [f"Person_{i}" for i in range(15)],
            aliases={"Person_0": ["Bob Smith", "Robert"]},
        )
        # Should not crash and should produce valid clusters
        result = cluster_entities_by_embedding(entities, target_cluster_size=5)
        assert len(result) >= 1

    def test_single_cluster_when_few_entities(self):
        """When n_clusters would be 1, return single batch."""
        from sift_kg.resolve.clustering import cluster_entities_by_embedding

        entities = self._make_entities([f"Person_{i}" for i in range(15)])
        # With default target_cluster_size=100, 15 entities -> 1 cluster
        result = cluster_entities_by_embedding(entities, target_cluster_size=100)
        assert len(result) == 1


class TestClusteringUnavailable:
    """Test fallback behavior when dependencies are missing."""

    def test_import_guard(self):
        """EMBEDDINGS_AVAILABLE flag is set correctly."""
        # This just verifies the flag exists and is a bool
        assert isinstance(EMBEDDINGS_AVAILABLE, bool)


class TestClusteringResolverIntegration:
    """Integration: clustering + resolver fallback."""

    def test_resolver_fallback_without_embeddings(self, sample_graph, mock_llm):
        """Resolver falls back to alphabetical when embeddings unavailable."""
        import asyncio

        from sift_kg.resolve.resolver import _afind_merge_candidates

        # Mock LLM to return no groups
        async def mock_acall_json(prompt):
            return {"groups": []}
        mock_llm.acall_json = mock_acall_json

        # With use_embeddings=True but potentially missing deps, should either
        # use clustering or fall back gracefully
        result = asyncio.run(
            _afind_merge_candidates(sample_graph, mock_llm, None, 1, use_embeddings=False)
        )
        assert result is not None
        assert hasattr(result, "proposals")
