"""Tests for --json output on CLI commands."""

import json

from typer.testing import CliRunner

from sift_kg.cli import app

runner = CliRunner()


class TestInfoJson:
    """Test sift info --json output."""

    def test_info_json_outputs_valid_json(self, tmp_dir, sample_extraction):
        """sift info --json produces valid JSON to stdout."""
        from sift_kg.graph.builder import build_graph

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")

        result = runner.invoke(app, ["info", "--json", "-o", str(tmp_dir)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "entities" in data
        assert "relations" in data
        assert "domain" in data

    def test_info_json_excludes_document_nodes(self, tmp_dir, sample_extraction):
        """Entity count in JSON excludes DOCUMENT nodes."""
        from sift_kg.graph.builder import build_graph

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")

        result = runner.invoke(app, ["info", "--json", "-o", str(tmp_dir)])
        data = json.loads(result.stdout)
        # sample_extraction has 3 entities (Alice, Acme, New York) + 1 DOCUMENT
        # JSON should only count the 3 substantive entities
        assert data["entities"] == 3

    def test_info_json_omits_missing_files(self, tmp_dir, sample_extraction):
        """Fields for missing files (merge_proposals, etc) are omitted."""
        from sift_kg.graph.builder import build_graph

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")

        result = runner.invoke(app, ["info", "--json", "-o", str(tmp_dir)])
        data = json.loads(result.stdout)
        assert "merge_proposals" not in data
        assert "relation_review" not in data


class TestSearchJson:
    """Test sift search --json output."""

    def test_search_json_outputs_valid_json(self, tmp_dir, sample_extraction):
        """sift search --json produces valid JSON."""
        from sift_kg.graph.builder import build_graph

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")

        result = runner.invoke(app, ["search", "Alice", "--json", "-o", str(tmp_dir)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "query" in data
        assert "results" in data
        assert len(data["results"]) >= 1

    def test_search_json_includes_relations(self, tmp_dir, sample_extraction):
        """sift search --json --relations includes relation data."""
        from sift_kg.graph.builder import build_graph

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")

        result = runner.invoke(
            app, ["search", "Alice", "--json", "--relations", "-o", str(tmp_dir)]
        )
        data = json.loads(result.stdout)
        assert len(data["results"]) >= 1
        assert "relations" in data["results"][0]

    def test_search_json_no_results(self, tmp_dir, sample_extraction):
        """sift search --json with no matches returns empty results."""
        from sift_kg.graph.builder import build_graph

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")

        result = runner.invoke(
            app, ["search", "nonexistent_xyz", "--json", "-o", str(tmp_dir)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["results"] == []

    def test_search_json_connections_excludes_metadata(self, tmp_dir, sample_extraction):
        """Connections count excludes MENTIONED_IN edges."""
        from sift_kg.graph.builder import build_graph

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")

        result = runner.invoke(app, ["search", "Alice", "--json", "-o", str(tmp_dir)])
        data = json.loads(result.stdout)
        alice = data["results"][0]
        # Alice has 1 substantive relation (WORKS_FOR -> Acme)
        # MENTIONED_IN edges to DOCUMENT nodes should be excluded
        assert alice["connections"] == 1


class TestTopologyCommand:
    """Test sift topology command."""

    def _build_graph_with_communities(self, tmp_dir, sample_extraction):
        """Helper: build graph and run community detection."""
        from sift_kg.graph.builder import build_graph
        from sift_kg.graph.communities import detect_communities, save_communities

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")
        communities = detect_communities(kg, min_community_size=1)
        if communities:
            save_communities(communities, tmp_dir)
        else:
            (tmp_dir / "communities.json").write_text("{}")
        return kg

    def test_topology_outputs_valid_json(self, tmp_dir, sample_extraction):
        """sift topology outputs valid JSON with expected schema."""
        self._build_graph_with_communities(tmp_dir, sample_extraction)

        result = runner.invoke(app, ["topology", "-o", str(tmp_dir)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "stats" in data
        assert "communities" in data
        assert "bridges" in data
        assert "isolated" in data
        assert "community_connections" in data

    def test_topology_stats_exclude_documents(self, tmp_dir, sample_extraction):
        """Stats exclude DOCUMENT nodes and MENTIONED_IN edges."""
        self._build_graph_with_communities(tmp_dir, sample_extraction)

        result = runner.invoke(app, ["topology", "-o", str(tmp_dir)])
        data = json.loads(result.stdout)
        # 3 substantive entities (Alice, Acme, New York)
        assert data["stats"]["entities"] == 3

    def test_topology_missing_graph_exits_1(self, tmp_dir):
        """Missing graph_data.json exits with code 1."""
        result = runner.invoke(app, ["topology", "-o", str(tmp_dir)])
        assert result.exit_code == 1

    def test_topology_missing_communities_exits_1(self, tmp_dir, sample_extraction):
        """Missing communities.json exits with code 1."""
        from sift_kg.graph.builder import build_graph

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")
        # Don't create communities.json

        result = runner.invoke(app, ["topology", "-o", str(tmp_dir)])
        assert result.exit_code == 1


class TestQueryCommand:
    """Test sift query command."""

    def _setup_graph(self, tmp_dir, sample_extraction):
        """Build graph with communities for query tests."""
        from sift_kg.graph.builder import build_graph
        from sift_kg.graph.communities import detect_communities, save_communities

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")
        communities = detect_communities(kg, min_community_size=1)
        if communities:
            save_communities(communities, tmp_dir)
        else:
            (tmp_dir / "communities.json").write_text("{}")
        return kg

    def test_query_by_name(self, tmp_dir, sample_extraction):
        """Query by name returns match with subgraph."""
        self._setup_graph(tmp_dir, sample_extraction)
        result = runner.invoke(app, ["query", "Alice", "-o", str(tmp_dir)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["match"] is not None
        assert data["match"]["name"] == "Alice Smith"
        assert len(data["subgraph"]["nodes"]) >= 1

    def test_query_by_exact_id(self, tmp_dir, sample_extraction):
        """Query by exact entity ID skips search."""
        self._setup_graph(tmp_dir, sample_extraction)
        result = runner.invoke(
            app, ["query", "person:alice_smith", "-o", str(tmp_dir)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["match"]["id"] == "person:alice_smith"

    def test_query_no_match(self, tmp_dir, sample_extraction):
        """No match returns null match."""
        self._setup_graph(tmp_dir, sample_extraction)
        result = runner.invoke(
            app, ["query", "nonexistent_xyz", "-o", str(tmp_dir)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["match"] is None

    def test_query_multiple_matches(self, tmp_dir):
        """Multiple matches returns top match + other_matches."""
        from sift_kg.extract.models import DocumentExtraction, ExtractedEntity

        entities = [
            ExtractedEntity(name="John Smith", entity_type="PERSON", confidence=0.9, context="ctx"),
            ExtractedEntity(name="Jane Smith", entity_type="PERSON", confidence=0.9, context="ctx"),
            ExtractedEntity(name="Acme Corp", entity_type="ORGANIZATION", confidence=0.9, context="ctx"),
        ]
        extraction = DocumentExtraction(
            document_id="test", document_path="/tmp/test.txt",
            entities=entities, relations=[], chunks_processed=1,
            model_used="test", cost_usd=0.0,
        )
        from sift_kg.graph.builder import build_graph

        kg = build_graph([extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")
        (tmp_dir / "communities.json").write_text("{}")

        result = runner.invoke(app, ["query", "Smith", "-o", str(tmp_dir)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["match"] is not None
        assert "other_matches" in data
        assert len(data["other_matches"]) >= 1
        assert "note" in data

    def test_query_missing_graph(self, tmp_dir):
        """Missing graph exits 1."""
        result = runner.invoke(app, ["query", "test", "-o", str(tmp_dir)])
        assert result.exit_code == 1

    def test_query_missing_communities_still_works(self, tmp_dir, sample_extraction):
        """Missing communities.json degrades gracefully."""
        from sift_kg.graph.builder import build_graph

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")

        result = runner.invoke(app, ["query", "Alice", "-o", str(tmp_dir)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["match"]["community"] is None

    def test_query_subgraph_excludes_documents(self, tmp_dir, sample_extraction):
        """Subgraph excludes DOCUMENT nodes."""
        self._setup_graph(tmp_dir, sample_extraction)
        result = runner.invoke(app, ["query", "Alice", "-o", str(tmp_dir)])
        data = json.loads(result.stdout)
        node_types = {n["entity_type"] for n in data["subgraph"]["nodes"]}
        assert "DOCUMENT" not in node_types

    def test_query_depth_flag(self, tmp_dir, sample_extraction):
        """--depth controls neighborhood size."""
        self._setup_graph(tmp_dir, sample_extraction)
        r1 = runner.invoke(app, ["query", "Alice", "--depth", "1", "-o", str(tmp_dir)])
        r2 = runner.invoke(app, ["query", "Alice", "--depth", "2", "-o", str(tmp_dir)])
        d1 = json.loads(r1.stdout)
        d2 = json.loads(r2.stdout)
        assert d2["depth"] == 2
        assert len(d2["subgraph"]["nodes"]) >= len(d1["subgraph"]["nodes"])
