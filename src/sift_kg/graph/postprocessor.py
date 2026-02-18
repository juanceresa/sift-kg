"""Graph post-processing for redundancy removal and noise pruning.

Implements Chilean KG paper (arXiv:2408.11975) cleanup rules:
- Activate passive relations (ENABLED_BY → ENABLES with edge flip)
- Remove self-loops
- Remove transitive redundant edges
- Prune isolated entities with no substantive connections
- Normalize undefined relation types to domain-defined types
"""

import logging
from typing import Any

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

# Common LLM-invented relation types → canonical mappings.
# Applied when a domain defines relation types and the LLM ignores constraints.
_RELATION_SYNONYMS: dict[str, str] = {
    "DEFENDS": "ASSOCIATED_WITH",
    "REPRESENTS": "ASSOCIATED_WITH",
    "DEFENDANT_IN": "PARTICIPATED_IN",
    "FRIEND_OF": "ASSOCIATED_WITH",
    "MARRIED_TO": "ASSOCIATED_WITH",
    "GAVE_TO": "ASSOCIATED_WITH",
    "SENT_TO": "ASSOCIATED_WITH",
    "TOLD_TO": "ASSOCIATED_WITH",
    "INVESTIGATED": "ASSOCIATED_WITH",
    "SUBPOENAED": "ASSOCIATED_WITH",
    "OWNED": "OWNS",
    "STAYED_AT": "RESIDED_AT",
    "REGISTERED_AT": "LOCATED_IN",
    "TRAVELED_FROM": "TRAVELED_TO",
}


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


def activate_passive_relations(
    kg: KnowledgeGraph, dry_run: bool = False
) -> dict[str, Any]:
    """Convert passive relation types to active form and flip edge direction.

    Passive relations like ENABLED_BY create counterintuitive arrow directions:
    mission_creep → ENABLED_BY → super_database means "mission creep is enabled
    by super-database", but the arrow points away from the enabler.

    This normalizes to active form: super_database → ENABLES → mission_creep,
    making arrows point from actor to target.

    Handles two patterns:
    - Explicit mapping for known passive types (e.g. GOVERNED_BY → GOVERNS)
    - Heuristic: *ED_BY suffix → strip to active form and flip

    Args:
        kg: KnowledgeGraph to clean (modified in place unless dry_run)
        dry_run: Report stats without modifying graph

    Returns:
        Stats dict with conversion counts
    """
    # Known passive → active mappings
    passive_to_active: dict[str, str] = {
        "ENABLED_BY": "ENABLES",
        "ENABLEDBY": "ENABLES",
        "GOVERNED_BY": "GOVERNS",
        "CAUSED_BY": "CAUSES",
        "FUNDED_BY": "FUNDS",
        "OWNED_BY": "OWNS",
        "EMPLOYED_BY": "EMPLOYS",
        "MANAGED_BY": "MANAGES",
        "CONTROLLED_BY": "CONTROLS",
        "REGULATED_BY": "REGULATES",
        "INFLUENCED_BY": "INFLUENCES",
        "CREATED_BY": "CREATES",
        "SUPPORTED_BY": "SUPPORTS",
        "OPPOSED_BY": "OPPOSES",
        "AUTHORIZED_BY": "AUTHORIZES",
        "CONSTRAINED_BY": "CONSTRAINS",
        "MONITORED_BY": "MONITORS",
        "ENFORCED_BY": "ENFORCES",
        "INVESTIGATED_BY": "INVESTIGATES",
        "OPERATED_BY": "OPERATES",
        "MAINTAINED_BY": "MAINTAINS",
    }

    import re

    stats: dict[str, Any] = {"passive_activated": 0, "passive_mappings": {}}
    to_flip: list[tuple[str, str, str, dict, str]] = []

    for source, target, key, data in list(kg.graph.edges(data=True, keys=True)):
        rel_type = data.get("relation_type", "")
        if not rel_type or rel_type in METADATA_RELATIONS:
            continue

        active = passive_to_active.get(rel_type)

        # Heuristic fallback: *ED_BY pattern
        if not active and re.match(r"^[A-Z]+ED_BY$", rel_type):
            # INVESTIGATED_BY → INVESTIGATES, MONITORED_BY → MONITORS
            stem = rel_type[:-3]  # strip "_BY"
            if stem.endswith("ED"):
                # Remove -ED, add -ES: ENABLED → ENABLES, GOVERNED → GOVERNS
                base = stem[:-2]
                if base.endswith("I"):
                    # STUDI_ED → STUDIES (not common in relation types)
                    active = base[:-1] + "IES"
                else:
                    active = base + "ES"
                    # Clean double-S: PROCESSSES → PROCESSES
                    if active.endswith("SES") and not active.endswith("SSES"):
                        pass  # keep as-is, e.g. CAUSES

        if not active:
            continue

        to_flip.append((source, target, key, data, active))
        stats["passive_mappings"][rel_type] = active

    if not dry_run:
        for source, target, key, data, active in to_flip:
            kg.graph.remove_edge(source, target, key=key)
            new_data = dict(data)
            new_data["relation_type"] = active
            kg.graph.add_edge(target, source, key=key, **new_data)

    stats["passive_activated"] = len(to_flip)

    if stats["passive_activated"]:
        unique = {f"{k} → {v}" for k, v in stats["passive_mappings"].items()}
        logger.info(
            f"Post-processing: activated {stats['passive_activated']} passive relations "
            f"({len(unique)} types: {', '.join(sorted(unique))})"
        )

    return stats


def normalize_relation_types(
    kg: KnowledgeGraph,
    domain_types: set[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Normalize undefined relation types to domain-defined types.

    Uses synonym mapping first, then falls back to ASSOCIATED_WITH for
    any relation type not in the domain's defined set.

    Args:
        kg: KnowledgeGraph to clean (modified in place unless dry_run)
        domain_types: Set of valid relation type names from domain config.
            If None, only applies synonym mapping without fallback.
        dry_run: Report stats without modifying graph

    Returns:
        Stats dict with normalization counts
    """
    stats: dict[str, Any] = {"normalized": 0, "mappings": {}}
    valid = domain_types or set()

    for source, target, key, data in list(kg.graph.edges(data=True, keys=True)):
        rel_type = data.get("relation_type", "")
        if not rel_type or rel_type in METADATA_RELATIONS:
            continue
        if valid and rel_type in valid:
            continue

        # Try synonym mapping first
        mapped = _RELATION_SYNONYMS.get(rel_type)
        if not mapped and valid:
            mapped = "ASSOCIATED_WITH"
        if not mapped:
            continue

        stats["normalized"] += 1
        stats["mappings"][rel_type] = mapped

        if not dry_run:
            kg.graph.edges[source, target, key]["relation_type"] = mapped

    if stats["normalized"]:
        unique = {f"{k} → {v}" for k, v in stats["mappings"].items()}
        logger.info(
            f"Post-processing: normalized {stats['normalized']} relations "
            f"({len(unique)} types: {', '.join(sorted(unique))})"
        )

    return stats


def fix_relation_directions(
    kg: KnowledgeGraph,
    relation_configs: dict[str, tuple[list[str], list[str], bool]],
    dry_run: bool = False,
) -> dict[str, int]:
    """Flip edges whose source/target entity types don't match the domain schema.

    When the LLM extracts "RESEARCHER PROPOSED_BY SYSTEM" instead of
    "SYSTEM PROPOSED_BY RESEARCHER", this detects the mismatch and flips
    the edge so the direction matches the domain's source_types/target_types.

    Args:
        kg: KnowledgeGraph to fix (modified in place unless dry_run)
        relation_configs: Map of relation_type → (source_types, target_types, symmetric).
            Only relations present in this map are checked.
        dry_run: Report stats without modifying graph

    Returns:
        Stats dict with flip/invalid counts
    """
    stats: dict[str, int] = {"relations_flipped": 0, "relations_invalid": 0}
    to_flip: list[tuple[str, str, str, dict]] = []

    for source, target, key, data in list(kg.graph.edges(data=True, keys=True)):
        rel_type = data.get("relation_type", "")
        if rel_type not in relation_configs or rel_type in METADATA_RELATIONS:
            continue

        source_types, target_types, symmetric = relation_configs[rel_type]
        if not source_types or not target_types:
            continue  # No type constraints defined
        if symmetric:
            continue  # Direction doesn't matter

        src_etype = kg.graph.nodes[source].get("entity_type", "")
        tgt_etype = kg.graph.nodes[target].get("entity_type", "")

        src_types_set = set(source_types)
        tgt_types_set = set(target_types)

        correct = src_etype in src_types_set and tgt_etype in tgt_types_set
        if correct:
            continue

        # Check if flipping would fix it
        flipped = tgt_etype in src_types_set and src_etype in tgt_types_set
        if flipped:
            to_flip.append((source, target, key, data))
        else:
            stats["relations_invalid"] += 1

    if not dry_run:
        for source, target, key, data in to_flip:
            kg.graph.remove_edge(source, target, key=key)
            kg.graph.add_edge(target, source, key=key, **data)

    stats["relations_flipped"] = len(to_flip)

    if stats["relations_flipped"]:
        logger.info(
            f"Post-processing: flipped {stats['relations_flipped']} relations "
            f"to match domain schema directions"
        )
    if stats["relations_invalid"]:
        logger.debug(
            f"Post-processing: {stats['relations_invalid']} relations have "
            f"entity types outside schema constraints (kept as-is)"
        )

    return stats
