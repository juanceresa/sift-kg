"""Tests for export formats with relation support fields."""

import csv
import sqlite3

from sift_kg.export import export_graph
from sift_kg.graph.knowledge_graph import KnowledgeGraph


def _build_supported_relation_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_entity("person:alice", "PERSON", "Alice")
    kg.add_entity("org:acme", "ORGANIZATION", "Acme")
    kg.add_relation(
        "r1", "person:alice", "org:acme", "WORKS_FOR",
        confidence=0.7, source_document="doc1", evidence="Mention one.",
    )
    kg.add_relation(
        "r2", "person:alice", "org:acme", "WORKS_FOR",
        confidence=0.6, source_document="doc2", evidence="Mention two.",
    )
    return kg


def test_csv_export_includes_support_columns(tmp_dir):
    """CSV relations export includes support and aggregated confidence fields."""
    kg = _build_supported_relation_graph()
    out_dir = tmp_dir / "csv_export"
    export_graph(kg, out_dir, "csv")

    with open(out_dir / "relations.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["support_count"] == "2"
    assert row["support_doc_count"] == "2"
    assert float(row["confidence"]) > 0.0
    assert "doc1" in row["support_documents"]
    assert "doc2" in row["support_documents"]


def test_sqlite_export_includes_support_columns(tmp_dir):
    """SQLite edge table includes support and aggregated confidence columns."""
    kg = _build_supported_relation_graph()
    db_path = tmp_dir / "graph.sqlite"
    export_graph(kg, db_path, "sqlite")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cols = cur.execute("PRAGMA table_info(edges)").fetchall()
    col_names = {col[1] for col in cols}
    row = cur.execute(
        "SELECT support_count, support_doc_count, confidence FROM edges"
    ).fetchone()
    conn.close()

    assert "support_count" in col_names
    assert "support_documents" in col_names
    assert "support_doc_count" in col_names
    assert "aggregated_confidence" not in col_names
    assert row[0] == 2
    assert row[1] == 2
    assert row[2] > 0.0
