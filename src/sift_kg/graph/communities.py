"""Community detection and structural analysis for knowledge graphs.

Provides Louvain-based community detection decoupled from the narrate
pipeline, plus topology analysis (bridges, isolation, inter-community
connections) for agent-consumable structural overviews.
"""

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from sift_kg.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def _build_clean_undirected(kg: KnowledgeGraph) -> nx.Graph:
    """Build undirected graph without DOCUMENT nodes and MENTIONED_IN edges.

    This is the standard graph preparation for community detection and
    topology analysis — strips metadata to expose substantive structure.
    """
    non_doc_nodes = [
        nid for nid, data in kg.graph.nodes(data=True)
        if data.get("entity_type") != "DOCUMENT"
    ]
    subgraph = kg.graph.subgraph(non_doc_nodes).copy()
    edges_to_remove = [
        (u, v, k) for u, v, k, d in subgraph.edges(keys=True, data=True)
        if d.get("relation_type") == "MENTIONED_IN"
    ]
    subgraph.remove_edges_from(edges_to_remove)
    return subgraph.to_undirected()


def detect_communities(
    kg: KnowledgeGraph,
    described_ids: set[str] | None = None,
    min_community_size: int = 8,
) -> list[list[dict[str, Any]]] | None:
    """Detect thematic communities using Louvain method.

    Args:
        kg: Knowledge graph to analyze.
        described_ids: If provided, only include entities with these IDs in
            community membership. If None, all non-DOCUMENT entities are
            included. Used by narrate to filter to described entities.
        min_community_size: Minimum members for a community to be included.

    Returns:
        List of communities (each a list of entity dicts with id, name,
        entity_type) sorted by total degree, or None if detection fails
        or produces <=1 community.
    """
    try:
        undirected = _build_clean_undirected(kg)
        raw_communities = nx.community.louvain_communities(undirected)
    except Exception as e:
        logger.debug(f"Community detection failed: {e}")
        return None

    if len(raw_communities) <= 1:
        return None

    # Build entity dicts for non-DOCUMENT nodes
    entity_map: dict[str, dict[str, Any]] = {}
    for nid, data in kg.graph.nodes(data=True):
        if data.get("entity_type") == "DOCUMENT":
            continue
        entity_map[nid] = {
            "id": nid,
            "name": data.get("name", nid),
            "entity_type": data.get("entity_type", "UNKNOWN"),
        }

    degree_map = dict(kg.graph.degree())

    result: list[list[dict[str, Any]]] = []
    for community_nodes in raw_communities:
        members = []
        for nid in community_nodes:
            if nid not in entity_map:
                continue
            if described_ids is not None and nid not in described_ids:
                continue
            members.append(entity_map[nid])
        if len(members) >= min_community_size:
            result.append(members)

    if not result:
        return None

    # Sort by total degree (most connected first)
    result.sort(
        key=lambda c: sum(degree_map.get(e["id"], 0) for e in c),
        reverse=True,
    )
    return result


def save_communities(
    communities: list[list[dict[str, Any]]],
    output_dir: Path,
    labels: dict[int, str] | None = None,
) -> Path:
    """Save community assignments to communities.json.

    Args:
        communities: List of communities (each a list of entity dicts).
        output_dir: Directory to write communities.json.
        labels: Optional map of community index → label string.
            If None, generates "Community 1", "Community 2", etc.

    Returns:
        Path to written communities.json.
    """
    comm_data: dict[str, str] = {}
    for i, community in enumerate(communities):
        label = (labels or {}).get(i, f"Community {i + 1}")
        for e in community:
            comm_data[e["id"]] = label

    comm_path = output_dir / "communities.json"
    comm_path.write_text(
        json.dumps(comm_data, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info(f"Communities saved ({len(set(comm_data.values()))} communities)")
    return comm_path


def load_communities(output_dir: Path) -> dict[str, str]:
    """Load community assignments from communities.json.

    Returns:
        Dict mapping entity_id → community label.
        Empty dict if file doesn't exist or is empty.
    """
    comm_path = output_dir / "communities.json"
    if not comm_path.exists():
        return {}
    data = json.loads(comm_path.read_text())
    return data if isinstance(data, dict) else {}


def load_communities_grouped(output_dir: Path) -> dict[str, list[str]]:
    """Load community assignments grouped by label.

    Inverts the entity_id → label mapping to label → list of entity_ids.

    Returns:
        Dict mapping community label → list of entity IDs.
        Empty dict if file doesn't exist.
    """
    flat = load_communities(output_dir)
    grouped: dict[str, list[str]] = {}
    for eid, label in flat.items():
        grouped.setdefault(label, []).append(eid)
    return grouped
