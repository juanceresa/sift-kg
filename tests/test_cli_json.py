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
