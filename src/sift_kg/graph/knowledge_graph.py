"""Knowledge graph using NetworkX MultiDiGraph."""

import json
import logging
from collections import Counter
from datetime import datetime
from importlib.metadata import version as _get_version
from math import prod
from pathlib import Path
from typing import Any

try:
    __version__ = _get_version("sift-kg")
except Exception:
    __version__ = "unknown"

import networkx as nx

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """NetworkX-based knowledge graph for entity-relation data."""

    VALID_CONFIDENCE_AGGREGATIONS = {"product_complement", "mean", "max"}

    def __init__(
        self,
        canonicalize_relations: bool = True,
        confidence_aggregation: str = "product_complement",
    ) -> None:
        self.graph = nx.MultiDiGraph()
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.canonicalize_relations = canonicalize_relations
        if confidence_aggregation not in self.VALID_CONFIDENCE_AGGREGATIONS:
            raise ValueError(
                f"Invalid confidence aggregation: {confidence_aggregation!r}. "
                f"Choose from: {', '.join(sorted(self.VALID_CONFIDENCE_AGGREGATIONS))}"
            )
        self.confidence_aggregation = confidence_aggregation

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
            relation_type = attrs.get("relation_type", "")
            canonical_key = attrs.get("canonical_key")
            relation_id = attrs.get("relation_id")
            edge_key = relation_id or canonical_key
            if edge_key is None and relation_type:
                edge_key = kg._canonical_relation_key(source, relation_type, target)
            if edge_key is not None:
                kg.graph.add_edge(source, target, key=edge_key, **attrs)
            else:
                edge_key = kg.graph.add_edge(source, target, **attrs)

            # Backward compatibility: old graphs had no support/mentions fields.
            edge_data = kg.graph.edges[source, target, edge_key]
            if relation_type:
                edge_data.setdefault(
                    "canonical_key",
                    kg._canonical_relation_key(source, relation_type, target),
                )
            kg._ensure_support_fields(edge_data, fallback_relation_id=str(edge_key))

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
        canonicalize: bool | None = None,
        confidence_aggregation: str | None = None,
    ) -> bool:
        """Add a relation edge. Returns False if source/target missing.

        If canonicalize=True (default), repeated mentions of the same
        source/relation/target triple are merged into one canonical edge.
        The ``confidence`` field becomes the aggregated confidence across
        all mentions (product-complement by default).
        """
        if not self.graph.has_node(source_id):
            logger.debug(f"Source entity {source_id} not found, skipping relation")
            return False
        if not self.graph.has_node(target_id):
            logger.debug(f"Target entity {target_id} not found, skipping relation")
            return False

        use_canonical = (
            self.canonicalize_relations if canonicalize is None else canonicalize
        )
        agg_method = confidence_aggregation or self.confidence_aggregation
        if agg_method not in self.VALID_CONFIDENCE_AGGREGATIONS:
            raise ValueError(
                f"Invalid confidence aggregation: {agg_method!r}. "
                f"Choose from: {', '.join(sorted(self.VALID_CONFIDENCE_AGGREGATIONS))}"
            )

        mention = {
            "source_document": source_document,
            "confidence": self._normalize_confidence(confidence),
            "evidence": evidence,
            "relation_id": relation_id,
        }

        if use_canonical:
            canonical_key = self._canonical_relation_key(
                source_id, relation_type, target_id
            )
            existing_keys = self._matching_relation_keys(
                source_id, target_id, relation_type
            )
            active_key: Any
            if self.graph.has_edge(source_id, target_id, key=canonical_key):
                active_key = canonical_key
            elif existing_keys:
                active_key = existing_keys[0]
            else:
                active_key = canonical_key

            if self.graph.has_edge(source_id, target_id, key=active_key):
                edge_data = self.graph.edges[source_id, target_id, active_key]
                self._ensure_support_fields(
                    edge_data,
                    fallback_relation_id=edge_data.get("relation_id", active_key),
                    aggregation_method=agg_method,
                )
                if active_key != canonical_key and not self.graph.has_edge(
                    source_id, target_id, key=canonical_key
                ):
                    migrated = dict(edge_data)
                    self.graph.remove_edge(source_id, target_id, key=active_key)
                    self.graph.add_edge(
                        source_id, target_id, key=canonical_key, **migrated
                    )
                    active_key = canonical_key
                    edge_data = self.graph.edges[source_id, target_id, active_key]
                elif active_key != canonical_key and self.graph.has_edge(
                    source_id, target_id, key=canonical_key
                ):
                    edge_data = self.graph.edges[source_id, target_id, canonical_key]
                    active_key = canonical_key

                mentions = edge_data.get("mentions", [])
                if not isinstance(mentions, list):
                    mentions = []
                mentions.append(mention)
                edge_data["mentions"] = mentions
                try:
                    current_support = int(edge_data.get("support_count", 0))
                except (TypeError, ValueError):
                    current_support = len(mentions) - 1
                edge_data["support_count"] = current_support + 1
                docs = edge_data.get("support_documents", [])
                if not isinstance(docs, list):
                    docs = []
                if source_document and source_document not in docs:
                    docs.append(source_document)
                edge_data["support_documents"] = docs
                edge_data["support_doc_count"] = len(docs)

                confidences = [
                    self._normalize_confidence(m.get("confidence", 0.5))
                    for m in mentions
                    if isinstance(m, dict)
                ]
                aggregated = self._aggregate_confidence(confidences, agg_method)
                edge_data["confidence"] = aggregated
                edge_data["canonical_key"] = canonical_key
                edge_data["relation_id"] = canonical_key
                # Update evidence/source_document if new mention has higher confidence
                best_so_far = edge_data.get("_best_mention_confidence", 0.0)
                if mention["confidence"] >= self._normalize_confidence(best_so_far):
                    edge_data["_best_mention_confidence"] = mention["confidence"]
                    edge_data["evidence"] = evidence
                    edge_data["source_document"] = source_document
            else:
                self.graph.add_edge(
                    source_id,
                    target_id,
                    key=canonical_key,
                    relation_id=canonical_key,
                    canonical_key=canonical_key,
                    relation_type=relation_type,
                    confidence=self._normalize_confidence(confidence),
                    evidence=evidence,
                    source_document=source_document,
                    support_count=1,
                    support_documents=[source_document] if source_document else [],
                    support_doc_count=1 if source_document else 0,
                    mentions=[mention],
                    _best_mention_confidence=self._normalize_confidence(confidence),
                )
        else:
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

    def export(self, include_mentions: bool = True) -> dict[str, Any]:
        """Export graph as JSON-serializable dict."""
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({"id": node_id, **data})

        links = []
        for source, target, _key, data in self.graph.edges(data=True, keys=True):
            # Strip underscore-prefixed internal tracking fields
            edge_data = {k: v for k, v in data.items() if not k.startswith("_")}
            if not include_mentions:
                edge_data.pop("mentions", None)
            links.append({"source": source, "target": target, **edge_data})

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
            "sift_kg_version": __version__,
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

    @staticmethod
    def _canonical_relation_key(source_id: str, relation_type: str, target_id: str) -> str:
        return f"{source_id}|{relation_type}|{target_id}"

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        try:
            conf = float(value)
        except (TypeError, ValueError):
            conf = 0.5
        return max(0.0, min(1.0, conf))

    @staticmethod
    def _aggregate_confidence(confidences: list[float], method: str) -> float:
        if not confidences:
            return 0.0
        if method == "mean":
            return sum(confidences) / len(confidences)
        if method == "max":
            return max(confidences)
        # product_complement: independent weak signals reinforce one another
        return 1.0 - prod(1.0 - c for c in confidences)

    def _matching_relation_keys(
        self, source_id: str, target_id: str, relation_type: str
    ) -> list[Any]:
        edge_data = self.graph.get_edge_data(source_id, target_id, default={})
        keys: list[Any] = []
        for key, data in edge_data.items():
            if data.get("relation_type") == relation_type:
                keys.append(key)
        return keys

    def _ensure_support_fields(
        self,
        edge_data: dict[str, Any],
        fallback_relation_id: str,
        aggregation_method: str | None = None,
    ) -> None:
        mentions_raw = edge_data.get("mentions")
        mentions: list[dict[str, Any]] = []
        if isinstance(mentions_raw, list):
            for mention in mentions_raw:
                if not isinstance(mention, dict):
                    continue
                mentions.append({
                    "source_document": mention.get("source_document", ""),
                    "confidence": self._normalize_confidence(mention.get("confidence", 0.5)),
                    "evidence": mention.get("evidence", ""),
                    "relation_id": mention.get("relation_id", fallback_relation_id),
                })

        if not mentions:
            mentions = [{
                "source_document": edge_data.get("source_document", ""),
                "confidence": self._normalize_confidence(edge_data.get("confidence", 0.5)),
                "evidence": edge_data.get("evidence", ""),
                "relation_id": edge_data.get("relation_id", fallback_relation_id),
            }]

        # Trust mentions as source of truth; ignore stale support_count
        support_count = len(mentions)

        support_docs = edge_data.get("support_documents")
        if not isinstance(support_docs, list):
            support_docs = []
        for mention in mentions:
            doc_id = mention.get("source_document", "")
            if doc_id and doc_id not in support_docs:
                support_docs.append(doc_id)

        confidences = [m["confidence"] for m in mentions]
        method = aggregation_method or self.confidence_aggregation
        aggregated = self._aggregate_confidence(confidences, method)
        best = max(mentions, key=lambda m: m["confidence"]) if mentions else None

        edge_data["mentions"] = mentions
        edge_data["support_count"] = support_count
        edge_data["support_documents"] = support_docs
        edge_data["support_doc_count"] = len(support_docs)
        edge_data["confidence"] = aggregated

        if best:
            edge_data["evidence"] = best.get("evidence", edge_data.get("evidence", ""))
            edge_data["source_document"] = best.get(
                "source_document",
                edge_data.get("source_document", ""),
            )
            edge_data["_best_mention_confidence"] = best.get("confidence", 0.0)
