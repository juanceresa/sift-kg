"""Export knowledge graphs to various formats.

Supports GraphML, GEXF (Gephi), CSV, and SQLite. All formats flatten
complex attributes (lists, dicts) to strings for compatibility.
"""

import csv
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import networkx as nx

from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.graph.postprocessor import strip_metadata
from sift_kg.visualize import EDGE_PALETTE, _color_for_entity

logger = logging.getLogger(__name__)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to (r, g, b) tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

SUPPORTED_FORMATS = ("json", "graphml", "gexf", "csv", "sqlite")


def export_graph(
    kg: KnowledgeGraph,
    output_path: Path,
    fmt: str,
    descriptions: dict[str, str] | None = None,
) -> Path:
    """Export a knowledge graph to the specified format.

    Args:
        kg: Knowledge graph to export
        output_path: Where to write the output (file or directory for CSV)
        fmt: Format string — "json", "graphml", "gexf", or "csv"
        descriptions: Optional entity descriptions to merge into node attributes

    Returns:
        Path to the written file (or directory for CSV)
    """
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}. Supported: {', '.join(SUPPORTED_FORMATS)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        return _export_json(kg, output_path)

    # Strip DOCUMENT nodes + MENTIONED_IN edges for visual/analytical exports
    clean = strip_metadata(kg)

    if fmt == "graphml":
        return _export_graphml(clean, output_path, descriptions)
    elif fmt == "gexf":
        return _export_gexf(clean, output_path, descriptions)
    elif fmt == "csv":
        return _export_csv(clean, output_path, descriptions)
    elif fmt == "sqlite":
        return _export_sqlite(clean, output_path, descriptions)
    raise ValueError(f"Unsupported format: {fmt}")


def _export_json(kg: KnowledgeGraph, output_path: Path) -> Path:
    """Export as sift-kg native JSON."""
    kg.save(output_path)
    return output_path


def _flatten_value(value: Any) -> str | int | float | bool:
    """Flatten complex values to strings for GraphML/GEXF compatibility."""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if any(isinstance(v, (dict, list, tuple, set)) for v in value):
            return json.dumps(value, default=str)
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return str(value)


def _coerce_support_docs(value: Any) -> list[str]:
    """Normalize support_documents from list/str into a clean list."""
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        if ";" in value:
            return [part.strip() for part in value.split(";") if part.strip()]
        if value.strip():
            return [value.strip()]
    return []


def _coerce_support_count(value: Any) -> int:
    """Normalize support_count to a positive int fallback."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 1
    return max(1, parsed)


def _build_flat_graph(
    kg: KnowledgeGraph,
    descriptions: dict[str, str] | None = None,
) -> nx.DiGraph:
    """Build a simple DiGraph with flattened attributes.

    GraphML/GEXF don't support MultiDiGraph well (parallel edges get
    collapsed). We merge parallel edges by concatenating relation types.

    Sets `label` on nodes and edges for Gephi/yEd display.
    Optionally merges entity descriptions from narrate output.
    """
    flat = nx.DiGraph()
    descriptions = descriptions or {}
    degrees = dict(kg.graph.degree())
    entity_color_map: dict[str, str] = {}

    for node_id, data in kg.graph.nodes(data=True):
        flat_attrs = {k: _flatten_value(v) for k, v in data.items()}
        # Gephi uses 'label' for display — set it to entity name
        flat_attrs["label"] = data.get("name", node_id)
        # Color by entity type — same palette as pyvis viewer
        entity_type = data.get("entity_type", "UNKNOWN")
        color = _color_for_entity(entity_type, entity_color_map)
        r, g, b = _hex_to_rgb(color)
        flat_attrs["color"] = color
        flat_attrs["r"] = r
        flat_attrs["g"] = g
        flat_attrs["b"] = b
        # Size by degree — same formula as pyvis viewer
        degree = degrees.get(node_id, 0)
        flat_attrs["size"] = float(max(8, min(50, 6 + degree * 2.5)))
        # Merge description if available
        if node_id in descriptions:
            flat_attrs["description"] = descriptions[node_id]
        flat.add_node(node_id, **flat_attrs)

    # Merge parallel edges between same source/target
    rel_color_map: dict[str, str] = {}
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    for source, target, _key, data in kg.graph.edges(data=True, keys=True):
        pair = (source, target)
        support_count = _coerce_support_count(data.get("support_count", 1))
        support_docs = set(_coerce_support_docs(data.get("support_documents", [])))
        if pair not in edge_map:
            edge_map[pair] = {k: _flatten_value(v) for k, v in data.items()}
            # Track multi-valued fields as sets for proper merging
            edge_map[pair]["_relation_types"] = {data.get("relation_type", "")}
            edge_map[pair]["_evidences"] = {data.get("evidence", "")} - {""}
            edge_map[pair]["_support_count"] = support_count
            edge_map[pair]["_support_docs"] = support_docs
        else:
            # Merge relation types (set-based, no substring issues)
            edge_map[pair]["_relation_types"].add(data.get("relation_type", ""))
            edge_map[pair]["_evidences"].add(data.get("evidence", ""))
            edge_map[pair]["_support_count"] += support_count
            edge_map[pair]["_support_docs"].update(support_docs)
            # Keep highest confidence
            new_conf = data.get("confidence", 0)
            if isinstance(new_conf, (int, float)) and new_conf > edge_map[pair].get("confidence", 0):
                edge_map[pair]["confidence"] = new_conf

    # Flatten merged sets back to strings + assign colors
    for attrs in edge_map.values():
        types = attrs.pop("_relation_types", set())
        rel_type = "; ".join(sorted(t for t in types if t))
        attrs["relation_type"] = rel_type
        evidences = attrs.pop("_evidences", set())
        if evidences:
            attrs["evidence"] = " | ".join(sorted(evidences))
        support_docs = attrs.pop("_support_docs", set())
        support_count = attrs.pop("_support_count", 1)
        attrs["support_count"] = int(support_count)
        attrs["support_documents"] = "; ".join(sorted(support_docs))
        attrs["support_doc_count"] = len(support_docs)
        # Color by relation type — same palette as pyvis viewer
        primary_type = next((t for t in sorted(types) if t), rel_type)
        if primary_type not in rel_color_map:
            idx = len(rel_color_map) % len(EDGE_PALETTE)
            rel_color_map[primary_type] = EDGE_PALETTE[idx]
        color = rel_color_map[primary_type]
        r, g, b = _hex_to_rgb(color)
        attrs["color"] = color
        attrs["r"] = r
        attrs["g"] = g
        attrs["b"] = b

    for (source, target), attrs in edge_map.items():
        # Gephi uses 'label' for edge display
        attrs["label"] = attrs.get("relation_type", "")
        flat.add_edge(source, target, **attrs)

    # Pre-compute layout so Gephi/yEd render a spread-out graph on import
    pos = nx.spring_layout(flat, k=3.0, iterations=100, seed=42, scale=1000)
    for node_id, (x, y) in pos.items():
        flat.nodes[node_id]["x"] = float(x)
        flat.nodes[node_id]["y"] = float(y)

    return flat


def _export_graphml(
    kg: KnowledgeGraph, output_path: Path, descriptions: dict[str, str] | None = None,
) -> Path:
    """Export as GraphML (compatible with yEd, Gephi, Cytoscape)."""
    flat = _build_flat_graph(kg, descriptions)
    nx.write_graphml(flat, str(output_path))
    logger.info(f"GraphML exported: {flat.number_of_nodes()} nodes, {flat.number_of_edges()} edges -> {output_path}")
    return output_path


def _export_gexf(
    kg: KnowledgeGraph, output_path: Path, descriptions: dict[str, str] | None = None,
) -> Path:
    """Export as GEXF (Gephi native format)."""
    flat = _build_flat_graph(kg, descriptions)
    nx.write_gexf(flat, str(output_path))
    logger.info(f"GEXF exported: {flat.number_of_nodes()} nodes, {flat.number_of_edges()} edges -> {output_path}")
    return output_path


def _export_csv(
    kg: KnowledgeGraph, output_dir: Path, descriptions: dict[str, str] | None = None,
) -> Path:
    """Export as CSV (entities.csv + relations.csv)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptions = descriptions or {}

    # Entities
    entities_path = output_dir / "entities.csv"
    entity_rows = []
    for node_id, data in kg.graph.nodes(data=True):
        entity_rows.append({
            "id": node_id,
            "name": data.get("name", ""),
            "entity_type": data.get("entity_type", ""),
            "confidence": data.get("confidence", ""),
            "source_documents": "; ".join(data.get("source_documents", [])),
            "attributes": json.dumps(data.get("attributes", {}), default=str),
            "description": descriptions.get(node_id, ""),
        })

    entity_fields = ["id", "name", "entity_type", "confidence", "source_documents", "attributes", "description"]
    with open(entities_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=entity_fields)
        writer.writeheader()
        writer.writerows(entity_rows)

    # Relations
    relations_path = output_dir / "relations.csv"
    relation_rows = []
    for source, target, _key, data in kg.graph.edges(data=True, keys=True):
        support_docs = _coerce_support_docs(data.get("support_documents", []))
        support_count = _coerce_support_count(data.get("support_count", 1))
        relation_rows.append({
            "source": source,
            "target": target,
            "relation_type": data.get("relation_type", ""),
            "confidence": data.get("confidence", ""),
            "support_count": support_count,
            "support_documents": "; ".join(support_docs),
            "support_doc_count": len(set(support_docs)),
            "evidence": data.get("evidence", ""),
            "source_document": data.get("source_document", ""),
        })

    relation_fields = [
        "source",
        "target",
        "relation_type",
        "confidence",
        "support_count",
        "support_documents",
        "support_doc_count",
        "evidence",
        "source_document",
    ]
    with open(relations_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=relation_fields)
        writer.writeheader()
        writer.writerows(relation_rows)

    logger.info(
        f"CSV exported: {len(entity_rows)} entities, {len(relation_rows)} relations -> {output_dir}"
    )
    return output_dir


def _export_sqlite(
    kg: KnowledgeGraph, output_path: Path, descriptions: dict[str, str] | None = None,
) -> Path:
    """Export as SQLite database (nodes + edges tables)."""
    descriptions = descriptions or {}

    # Remove existing file so we get a clean export
    if output_path.exists():
        output_path.unlink()

    conn = sqlite3.connect(str(output_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY,
            name TEXT,
            entity_type TEXT,
            confidence REAL,
            source_documents TEXT,
            attributes TEXT,
            description TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE edges (
            source_id TEXT,
            target_id TEXT,
            relation_type TEXT,
            confidence REAL,
            support_count INTEGER,
            support_documents TEXT,
            support_doc_count INTEGER,
            evidence TEXT,
            source_document TEXT,
            FOREIGN KEY(source_id) REFERENCES nodes(node_id),
            FOREIGN KEY(target_id) REFERENCES nodes(node_id)
        )
    """)
    cur.execute("CREATE INDEX idx_edges_source ON edges(source_id)")
    cur.execute("CREATE INDEX idx_edges_target ON edges(target_id)")
    cur.execute("CREATE INDEX idx_edges_relation ON edges(relation_type)")

    for node_id, data in kg.graph.nodes(data=True):
        cur.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                node_id,
                data.get("name", ""),
                data.get("entity_type", ""),
                data.get("confidence"),
                "; ".join(data.get("source_documents", [])),
                json.dumps(data.get("attributes", {}), default=str),
                descriptions.get(node_id, ""),
            ),
        )

    for source, target, _key, data in kg.graph.edges(data=True, keys=True):
        support_docs = _coerce_support_docs(data.get("support_documents", []))
        support_count = _coerce_support_count(data.get("support_count", 1))
        cur.execute(
            "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source,
                target,
                data.get("relation_type", ""),
                data.get("confidence"),
                support_count,
                "; ".join(support_docs),
                len(set(support_docs)),
                data.get("evidence", ""),
                data.get("source_document", ""),
            ),
        )

    conn.commit()
    node_count = cur.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edge_count = cur.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    conn.close()

    logger.info(f"SQLite exported: {node_count} nodes, {edge_count} edges -> {output_path}")
    return output_path
