"""Graph post-processing for redundancy removal.

Implements Chilean KG paper (arXiv:2408.11975) cleanup rules:
- Remove self-loops
- Remove transitive redundant edges
"""

import logging

import networkx as nx

from sift_kg.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Relation types where transitivity is safe to assume.
# LOCATED_IN: if A in B, B in C, then A in C is redundant.
# Deliberately excludes EMPLOYED_BY (not always transitive).
TRANSITIVE_RELATIONS = {"LOCATED_IN"}


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
