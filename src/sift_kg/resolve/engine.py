"""Graph surgery engine — apply confirmed merges and relation rejections."""

import logging

from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.resolve.models import MergeFile, RelationReviewFile

logger = logging.getLogger(__name__)


def apply_merges(kg: KnowledgeGraph, merge_file: MergeFile) -> dict[str, int]:
    """Apply confirmed merge proposals to the knowledge graph.

    For each confirmed proposal:
    1. Merge member node data into canonical node
    2. Rewrite all edges pointing to/from members to point to canonical
    3. Remove member nodes
    4. Remove self-loops created by merging

    Args:
        kg: KnowledgeGraph to modify in place
        merge_file: MergeFile with proposals (only CONFIRMED are applied)

    Returns:
        Stats dict with counts
    """
    confirmed = merge_file.confirmed
    if not confirmed:
        logger.info("No confirmed merges to apply")
        return {"merges_applied": 0, "nodes_removed": 0, "self_loops_removed": 0}

    stats = {"merges_applied": 0, "nodes_removed": 0, "self_loops_removed": 0}

    # Build full merge map: member_id → canonical_id
    merge_map: dict[str, str] = {}
    for proposal in confirmed:
        for member in proposal.members:
            if member.id != proposal.canonical_id:
                merge_map[member.id] = proposal.canonical_id

    # Validate: skip members/canonicals not in graph
    valid_map: dict[str, str] = {}
    for member_id, canonical_id in merge_map.items():
        if not kg.graph.has_node(member_id):
            logger.debug(f"Member {member_id} not in graph, skipping")
            continue
        if not kg.graph.has_node(canonical_id):
            logger.warning(f"Canonical {canonical_id} not in graph, skipping merge")
            continue
        valid_map[member_id] = canonical_id

    if not valid_map:
        logger.info("No valid merges after validation")
        return stats

    # Merge node data (member into canonical)
    for member_id, canonical_id in valid_map.items():
        _merge_node_data(kg, canonical_id, member_id)

    # Rewrite edges
    edges_to_rewrite = list(kg.graph.edges(data=True, keys=True))
    for source, target, key, data in edges_to_rewrite:
        new_source = valid_map.get(source, source)
        new_target = valid_map.get(target, target)

        if new_source != source or new_target != target:
            # Remove old edge
            kg.graph.remove_edge(source, target, key=key)

            # Skip self-loops
            if new_source == new_target:
                stats["self_loops_removed"] += 1
                continue

            # Add rewritten edge
            kg.graph.add_edge(new_source, new_target, key=key, **data)

    # Remove merged nodes
    for member_id in valid_map:
        if kg.graph.has_node(member_id):
            kg.graph.remove_node(member_id)
            stats["nodes_removed"] += 1

    stats["merges_applied"] = len(valid_map)

    logger.info(
        f"Applied {stats['merges_applied']} merges: "
        f"{stats['nodes_removed']} nodes removed, "
        f"{stats['self_loops_removed']} self-loops dropped"
    )

    return stats


def _merge_node_data(kg: KnowledgeGraph, canonical_id: str, member_id: str) -> None:
    """Merge member node data into canonical, preserving canonical values."""
    canonical = kg.graph.nodes[canonical_id]
    member = kg.graph.nodes[member_id]

    # Extend source_documents
    canonical_docs = canonical.get("source_documents", [])
    member_docs = member.get("source_documents", [])
    for doc in member_docs:
        if doc not in canonical_docs:
            canonical_docs.append(doc)
    canonical["source_documents"] = canonical_docs

    # Keep higher confidence
    if member.get("confidence", 0) > canonical.get("confidence", 0):
        canonical["confidence"] = member["confidence"]

    # Merge attributes (canonical takes precedence)
    canonical_attrs = canonical.get("attributes", {})
    member_attrs = member.get("attributes", {})
    for key, value in member_attrs.items():
        if key not in canonical_attrs or canonical_attrs[key] is None:
            canonical_attrs[key] = value
    canonical["attributes"] = canonical_attrs


def apply_relation_rejections(
    kg: KnowledgeGraph, review_file: RelationReviewFile
) -> int:
    """Remove REJECTED relations from the graph.

    Args:
        kg: KnowledgeGraph to modify in place
        review_file: RelationReviewFile with reviewed relations

    Returns:
        Number of relations removed
    """
    rejected = review_file.rejected
    if not rejected:
        return 0

    # Build rejection keys: (source_id, target_id, relation_type)
    rejection_keys: set[tuple[str, str, str]] = set()
    for entry in rejected:
        rejection_keys.add((entry.source_id, entry.target_id, entry.relation_type))
        # Also handle symmetric — if A→B rejected, B→A should be too
        rejection_keys.add((entry.target_id, entry.source_id, entry.relation_type))

    # Find and remove matching edges
    edges_to_remove = []
    for source, target, key, data in kg.graph.edges(data=True, keys=True):
        rel_type = data.get("relation_type", "")
        if (source, target, rel_type) in rejection_keys:
            edges_to_remove.append((source, target, key))

    removed = 0
    for source, target, key in edges_to_remove:
        try:
            kg.graph.remove_edge(source, target, key=key)
            removed += 1
        except (KeyError, Exception) as e:
            logger.warning(f"Could not remove edge {source} -> {target} (key={key}): {e}")

    if removed:
        logger.info(f"Removed {removed} rejected relations")

    return removed
