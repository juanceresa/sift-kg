"""Knowledge graph using NetworkX MultiDiGraph."""

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """NetworkX-based knowledge graph for entity-relation data."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        """Load graph from JSON file."""
        data = json.loads(Path(path).read_text())
        kg = cls()

        metadata = data.get("metadata", {})
        created_at = metadata.get("created_at")
        if isinstance(created_at, str):
            try:
                kg.created_at = datetime.fromisoformat(created_at)
            except ValueError:
                pass

        for node in data.get("nodes", []):
            node_id = node.get("id")
            if node_id is None:
                continue
            attrs = {k: v for k, v in node.items() if k != "id"}
            kg.graph.add_node(node_id, **attrs)

        for link in data.get("links", data.get("edges", [])):
            source = link.get("source")
            target = link.get("target")
            if source is None or target is None:
                continue
            attrs = {k: v for k, v in link.items() if k not in ("source", "target")}
            relation_id = attrs.get("relation_id")
            if relation_id:
                kg.graph.add_edge(source, target, key=relation_id, **attrs)
            else:
                kg.graph.add_edge(source, target, **attrs)

        return kg

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        confidence: float = 0.5,
        source_documents: list[str] | None = None,
        **attrs: Any,
    ) -> None:
        """Add or update an entity node."""
        if self.graph.has_node(entity_id):
            # Merge: update confidence if higher, extend source_documents
            existing = self.graph.nodes[entity_id]
            if confidence > existing.get("confidence", 0):
                existing["confidence"] = confidence
            existing_docs = existing.get("source_documents", [])
            for doc in source_documents or []:
                if doc not in existing_docs:
                    existing_docs.append(doc)
            existing["source_documents"] = existing_docs
            # Merge attributes dict
            new_attributes = attrs.pop("attributes", {})
            if new_attributes:
                existing_attrs = existing.get("attributes", {})
                existing_attrs.update(new_attributes)
                existing["attributes"] = existing_attrs
            # Merge remaining kwargs (context, etc.)
            for key, value in attrs.items():
                if key not in existing or existing[key] in (None, "", []):
                    existing[key] = value
        else:
            self.graph.add_node(
                entity_id,
                entity_type=entity_type,
                name=name,
                confidence=confidence,
                source_documents=source_documents or [],
                **attrs,
            )
        self.updated_at = datetime.now()

    def add_relation(
        self,
        relation_id: str,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float = 0.5,
        evidence: str = "",
        source_document: str = "",
    ) -> bool:
        """Add a relation edge. Returns False if source/target missing."""
        if not self.graph.has_node(source_id):
            logger.debug(f"Source entity {source_id} not found, skipping relation")
            return False
        if not self.graph.has_node(target_id):
            logger.debug(f"Target entity {target_id} not found, skipping relation")
            return False

        self.graph.add_edge(
            source_id,
            target_id,
            key=relation_id,
            relation_id=relation_id,
            relation_type=relation_type,
            confidence=confidence,
            evidence=evidence,
            source_document=source_document,
        )
        self.updated_at = datetime.now()
        return True

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Get entity data by ID."""
        if not self.graph.has_node(entity_id):
            return None
        return dict(self.graph.nodes[entity_id])

    def get_relations(self, entity_id: str, direction: str = "both") -> list[dict[str, Any]]:
        """Get relations for an entity. Direction: 'in', 'out', or 'both'."""
        if not self.graph.has_node(entity_id):
            return []

        relations = []
        if direction in ("out", "both"):
            for src, tgt, _key, data in self.graph.out_edges(entity_id, keys=True, data=True):
                relations.append({"source": src, "target": tgt, **data})

        if direction in ("in", "both"):
            for src, tgt, _key, data in self.graph.in_edges(entity_id, keys=True, data=True):
                relations.append({"source": src, "target": tgt, **data})

        return relations

    def export(self) -> dict[str, Any]:
        """Export graph as JSON-serializable dict."""
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({"id": node_id, **data})

        links = []
        for source, target, _key, data in self.graph.edges(data=True, keys=True):
            links.append({"source": source, "target": target, **data})

        # Build metadata
        type_counts = Counter(
            data.get("entity_type") for _, data in self.graph.nodes(data=True)
        )

        metadata = {
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "entity_count": self.graph.number_of_nodes(),
            "relation_count": self.graph.number_of_edges(),
            "document_count": type_counts.get("DOCUMENT", 0),
            "entity_type_summary": dict(type_counts),
            "sift_kg_version": "0.2.0",
        }

        return {"metadata": metadata, "nodes": nodes, "links": links}

    def save(self, path: str | Path) -> None:
        """Save graph to JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.export(), indent=2, default=str))
        logger.info(
            f"Graph saved: {self.graph.number_of_nodes()} entities, "
            f"{self.graph.number_of_edges()} relations → {out}"
        )

    @property
    def entity_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def relation_count(self) -> int:
        return self.graph.number_of_edges()
