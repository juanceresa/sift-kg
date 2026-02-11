"""Export knowledge graphs to various formats.

Supports GraphML, GEXF (Gephi), and CSV. All formats flatten
complex attributes (lists, dicts) to strings for compatibility.
"""

import csv
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import networkx as nx

from sift_kg.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("json", "graphml", "gexf", "csv")


def export_graph(kg: KnowledgeGraph, output_path: Path, fmt: str) -> Path:
    """Export a knowledge graph to the specified format.

    Args:
        kg: Knowledge graph to export
        output_path: Where to write the output (file or directory for CSV)
        fmt: Format string — "json", "graphml", "gexf", or "csv"

    Returns:
        Path to the written file (or directory for CSV)
    """
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}. Supported: {', '.join(SUPPORTED_FORMATS)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        return _export_json(kg, output_path)
    elif fmt == "graphml":
        return _export_graphml(kg, output_path)
    elif fmt == "gexf":
        return _export_gexf(kg, output_path)
    elif fmt == "csv":
        return _export_csv(kg, output_path)
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
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return str(value)


def _build_flat_graph(kg: KnowledgeGraph) -> nx.DiGraph:
    """Build a simple DiGraph with flattened attributes.

    GraphML/GEXF don't support MultiDiGraph well (parallel edges get
    collapsed). We merge parallel edges by concatenating relation types.
    """
    flat = nx.DiGraph()

    for node_id, data in kg.graph.nodes(data=True):
        flat_attrs = {k: _flatten_value(v) for k, v in data.items()}
        flat.add_node(node_id, **flat_attrs)

    # Merge parallel edges between same source/target
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    for source, target, _key, data in kg.graph.edges(data=True, keys=True):
        pair = (source, target)
        if pair not in edge_map:
            edge_map[pair] = {k: _flatten_value(v) for k, v in data.items()}
        else:
            # Append relation type if different
            existing_type = edge_map[pair].get("relation_type", "")
            new_type = data.get("relation_type", "")
            if new_type and new_type not in existing_type:
                edge_map[pair]["relation_type"] = f"{existing_type}; {new_type}"

    for (source, target), attrs in edge_map.items():
        flat.add_edge(source, target, **attrs)

    return flat


def _export_graphml(kg: KnowledgeGraph, output_path: Path) -> Path:
    """Export as GraphML (compatible with yEd, Gephi, Cytoscape)."""
    flat = _build_flat_graph(kg)
    nx.write_graphml(flat, str(output_path))
    logger.info(f"GraphML exported: {flat.number_of_nodes()} nodes, {flat.number_of_edges()} edges -> {output_path}")
    return output_path


def _export_gexf(kg: KnowledgeGraph, output_path: Path) -> Path:
    """Export as GEXF (Gephi native format)."""
    flat = _build_flat_graph(kg)
    nx.write_gexf(flat, str(output_path))
    logger.info(f"GEXF exported: {flat.number_of_nodes()} nodes, {flat.number_of_edges()} edges -> {output_path}")
    return output_path


def _export_csv(kg: KnowledgeGraph, output_dir: Path) -> Path:
    """Export as CSV (entities.csv + relations.csv)."""
    output_dir.mkdir(parents=True, exist_ok=True)

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
        })

    if entity_rows:
        fieldnames = ["id", "name", "entity_type", "confidence", "source_documents", "attributes"]
        with open(entities_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entity_rows)

    # Relations
    relations_path = output_dir / "relations.csv"
    relation_rows = []
    for source, target, _key, data in kg.graph.edges(data=True, keys=True):
        relation_rows.append({
            "source": source,
            "target": target,
            "relation_type": data.get("relation_type", ""),
            "confidence": data.get("confidence", ""),
            "evidence": data.get("evidence", ""),
            "source_document": data.get("source_document", ""),
        })

    if relation_rows:
        fieldnames = ["source", "target", "relation_type", "confidence", "evidence", "source_document"]
        with open(relations_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(relation_rows)

    logger.info(
        f"CSV exported: {len(entity_rows)} entities, {len(relation_rows)} relations -> {output_dir}"
    )
    return output_dir
