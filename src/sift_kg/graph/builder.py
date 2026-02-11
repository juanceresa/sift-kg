"""Build knowledge graph from extraction results.

Entity resolution happens in a later phase (resolve/).
"""

import json
import logging
from pathlib import Path

from unidecode import unidecode

from sift_kg.extract.models import DocumentExtraction
from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.graph.postprocessor import prune_isolated_entities, remove_redundant_edges
from sift_kg.graph.prededup import prededup_entities

logger = logging.getLogger(__name__)


def _make_entity_id(name: str, entity_type: str) -> str:
    """Generate a stable entity ID from name + type.

    Normalizes to lowercase ASCII for consistent matching.
    """
    normalized = unidecode(name.lower().strip())
    # Replace spaces/special chars with underscores
    normalized = "".join(c if c.isalnum() else "_" for c in normalized)
    # Collapse multiple underscores
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    return f"{entity_type.lower()}:{normalized}"


def _make_relation_id(
    source_id: str, target_id: str, relation_type: str, doc_id: str
) -> str:
    """Generate a stable relation ID."""
    return f"{source_id}|{relation_type}|{target_id}|{doc_id}"


def build_graph(
    extractions: list[DocumentExtraction],
    postprocess: bool = True,
) -> KnowledgeGraph:
    """Build knowledge graph from extraction results.

    Args:
        extractions: List of document extractions
        postprocess: Whether to remove redundant edges

    Returns:
        Populated KnowledgeGraph
    """
    kg = KnowledgeGraph()
    stats = {
        "documents": 0,
        "entities_added": 0,
        "relations_added": 0,
        "relations_skipped": 0,
    }

    # Entity name → ID lookup (for resolving relation endpoints)
    name_to_id: dict[str, str] = {}

    # Pre-dedup: merge near-identical entity names deterministically
    canonical_map = prededup_entities(extractions)

    for extraction in extractions:
        if extraction.error:
            logger.warning(f"Skipping {extraction.document_id} (error: {extraction.error})")
            continue

        doc_id = extraction.document_id
        stats["documents"] += 1

        # Add DOCUMENT entity
        kg.add_entity(
            entity_id=f"doc:{doc_id}",
            entity_type="DOCUMENT",
            name=doc_id,
            confidence=1.0,
            source_documents=[doc_id],
            attributes={"path": extraction.document_path},
        )

        # Add entities
        for entity in extraction.entities:
            entity_name = canonical_map.get(
                (entity.entity_type, entity.name), entity.name
            )
            eid = _make_entity_id(entity_name, entity.entity_type)
            kg.add_entity(
                entity_id=eid,
                entity_type=entity.entity_type,
                name=entity_name,
                confidence=entity.confidence,
                source_documents=[doc_id],
                attributes=entity.attributes,
                context=entity.context,
            )
            stats["entities_added"] += 1

            # Track name → ID for relation resolution (both original and canonical)
            name_to_id[entity.name.lower().strip()] = eid
            name_to_id[entity_name.lower().strip()] = eid

            # Add MENTIONED_IN relation to document
            kg.add_relation(
                relation_id=f"{eid}|MENTIONED_IN|doc:{doc_id}",
                source_id=eid,
                target_id=f"doc:{doc_id}",
                relation_type="MENTIONED_IN",
                confidence=entity.confidence,
                source_document=doc_id,
            )

        # Add relations
        for rel in extraction.relations:
            source_id = _resolve_entity_name(rel.source_entity, name_to_id)
            target_id = _resolve_entity_name(rel.target_entity, name_to_id)

            if not source_id or not target_id:
                logger.debug(
                    f"Skipping relation {rel.relation_type}: "
                    f"unresolved entity ({rel.source_entity} → {rel.target_entity})"
                )
                stats["relations_skipped"] += 1
                continue

            rid = _make_relation_id(source_id, target_id, rel.relation_type, doc_id)
            added = kg.add_relation(
                relation_id=rid,
                source_id=source_id,
                target_id=target_id,
                relation_type=rel.relation_type,
                confidence=rel.confidence,
                evidence=rel.evidence,
                source_document=doc_id,
            )
            if added:
                stats["relations_added"] += 1
            else:
                stats["relations_skipped"] += 1

    # Post-process
    if postprocess and kg.entity_count > 0:
        cleanup = remove_redundant_edges(kg)
        stats.update(cleanup)
        prune = prune_isolated_entities(kg)
        stats.update(prune)

    logger.info(
        f"Graph built: {stats['documents']} docs → "
        f"{kg.entity_count} entities, {kg.relation_count} relations "
        f"({stats['relations_skipped']} skipped)"
    )

    return kg


def _resolve_entity_name(name: str, name_to_id: dict[str, str]) -> str | None:
    """Resolve an entity name to its graph ID.

    Tries exact match first, then case-insensitive.
    """
    key = name.lower().strip()
    if key in name_to_id:
        return name_to_id[key]
    return None


def flag_relations_for_review(
    kg: KnowledgeGraph,
    confidence_threshold: float = 0.7,
    review_types: set[str] | None = None,
) -> list[dict]:
    """Flag relations that need human review.

    Flags relations that are:
    - Below the confidence threshold, OR
    - Of a type marked review_required in domain config

    Args:
        kg: Knowledge graph with relations
        confidence_threshold: Flag relations below this confidence
        review_types: Relation types that always require review

    Returns:
        List of flagged relation dicts
    """
    flagged = []
    review_types = review_types or set()

    for source, target, _key, data in kg.graph.edges(data=True, keys=True):
        rel_type = data.get("relation_type", "")
        confidence = data.get("confidence", 0.5)

        # Skip MENTIONED_IN — these are auto-generated, not extracted
        if rel_type == "MENTIONED_IN":
            continue

        reasons = []
        if confidence < confidence_threshold:
            reasons.append(f"Low confidence ({confidence:.2f} < {confidence_threshold})")
        if rel_type in review_types:
            reasons.append(f"Type '{rel_type}' requires review")

        if reasons:
            source_data = kg.graph.nodes.get(source, {})
            target_data = kg.graph.nodes.get(target, {})
            flagged.append({
                "source_id": source,
                "source_name": source_data.get("name", source),
                "target_id": target,
                "target_name": target_data.get("name", target),
                "relation_type": rel_type,
                "confidence": confidence,
                "evidence": data.get("evidence", ""),
                "source_document": data.get("source_document", ""),
                "flag_reason": "; ".join(reasons),
            })

    return flagged


def load_extractions(output_dir: Path) -> list[DocumentExtraction]:
    """Load all extraction JSONs from output directory."""
    extractions_dir = output_dir / "extractions"
    if not extractions_dir.exists():
        return []

    results = []
    for path in sorted(extractions_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
            results.append(DocumentExtraction(**raw))
        except Exception as e:
            logger.warning(f"Failed to load {path.name}: {e}")

    return results
