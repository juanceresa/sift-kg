"""Graph post-processing for redundancy removal and noise pruning.

Implements Chilean KG paper (arXiv:2408.11975) cleanup rules:
- Remove self-loops
- Remove transitive redundant edges
- Prune isolated entities with no substantive connections
"""

import logging

import networkx as nx

from sift_kg.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Relation types where transitivity is safe to assume.
# LOCATED_IN: if A in B, B in C, then A in C is redundant.
# Deliberately excludes EMPLOYED_BY (not always transitive).
TRANSITIVE_RELATIONS = {"LOCATED_IN"}

# Edge types that don't count as substantive connections for pruning.
# MENTIONED_IN just links entities to their source documents.
METADATA_RELATIONS = {"MENTIONED_IN"}


def strip_metadata(kg: KnowledgeGraph) -> KnowledgeGraph:
    """Return a copy of the graph without DOCUMENT nodes and metadata edges.

    Removes DOCUMENT-type nodes and MENTIONED_IN edges which are provenance
    tracking, not substantive relationships. Used by export and visualization
    layers to produce cleaner output.
    """
    clean = KnowledgeGraph()
    clean.created_at = kg.created_at
    clean.updated_at = kg.updated_at

    for node_id, data in kg.graph.nodes(data=True):
        if data.get("entity_type") == "DOCUMENT":
            continue
        clean.graph.add_node(node_id, **data)

    for source, target, key, data in kg.graph.edges(data=True, keys=True):
        if data.get("relation_type") in METADATA_RELATIONS:
            continue
        if not clean.graph.has_node(source) or not clean.graph.has_node(target):
            continue
        clean.graph.add_edge(source, target, key=key, **data)

    logger.info(
        f"Stripped metadata: {kg.entity_count} → {clean.entity_count} entities, "
        f"{kg.relation_count} → {clean.relation_count} relations"
    )
    return clean


def remove_redundant_edges(
    kg: KnowledgeGraph, dry_run: bool = False
) -> dict[str, int]:
    """Remove self-loops and transitive redundant edges.

    Args:
        kg: KnowledgeGraph to clean (modified in place unless dry_run)
        dry_run: Report stats without modifying graph

    Returns:
        Stats dict with removal counts
    """
    stats = {"self_loops_removed": 0, "transitive_removed": 0, "edges_removed": 0}

    # Pass 1: self-loops
    self_loops = list(nx.selfloop_edges(kg.graph, keys=True))
    for u, v, key in self_loops:
        if not dry_run:
            kg.graph.remove_edge(u, v, key=key)
        stats["self_loops_removed"] += 1
        stats["edges_removed"] += 1

    # Pass 2: transitive redundancies
    for rel_type in TRANSITIVE_RELATIONS:
        typed_edges = [
            (u, v, k)
            for u, v, k, d in kg.graph.edges(data=True, keys=True)
            if d.get("relation_type") == rel_type
        ]

        # Build adjacency for this relation type
        adjacency: dict[str, set[str]] = {}
        for u, v, _k in typed_edges:
            adjacency.setdefault(u, set()).add(v)

        # For each A→C edge, check if A→B→C path exists
        for source, target, key in typed_edges:
            intermediates = adjacency.get(source, set())
            for mid in intermediates:
                if mid != target and target in adjacency.get(mid, set()):
                    # A→B→C exists, so A→C is redundant
                    try:
                        if not dry_run:
                            kg.graph.remove_edge(source, target, key=key)
                        stats["transitive_removed"] += 1
                        stats["edges_removed"] += 1
                    except nx.NetworkXError:
                        pass  # Already removed
                    break

    if stats["edges_removed"]:
        logger.info(
            f"Post-processing: removed {stats['edges_removed']} edges "
            f"({stats['self_loops_removed']} self-loops, "
            f"{stats['transitive_removed']} transitive)"
        )

    return stats


def prune_isolated_entities(
    kg: KnowledgeGraph, dry_run: bool = False
) -> dict[str, int]:
    """Remove entities with no substantive connections.

    An entity is considered isolated if all of its edges are metadata
    relations (e.g. MENTIONED_IN). These are extracted entities that
    never connected to anything meaningful in the graph.

    Document nodes are never pruned.

    Args:
        kg: KnowledgeGraph to clean (modified in place unless dry_run)
        dry_run: Report stats without modifying graph

    Returns:
        Stats dict with removal counts
    """
    stats = {"entities_pruned": 0}
    to_remove = []

    for node_id, data in kg.graph.nodes(data=True):
        if data.get("entity_type") == "DOCUMENT":
            continue

        has_substantive = False
        for _, _, edge_data in kg.graph.edges(node_id, data=True):
            if edge_data.get("relation_type") not in METADATA_RELATIONS:
                has_substantive = True
                break
        if has_substantive:
            continue
        for _, _, edge_data in kg.graph.in_edges(node_id, data=True):
            if edge_data.get("relation_type") not in METADATA_RELATIONS:
                has_substantive = True
                break

        if not has_substantive:
            to_remove.append(node_id)

    if not dry_run:
        for node_id in to_remove:
            kg.graph.remove_node(node_id)

    stats["entities_pruned"] = len(to_remove)

    if stats["entities_pruned"]:
        logger.info(
            f"Post-processing: pruned {stats['entities_pruned']} isolated entities"
        )

    return stats
