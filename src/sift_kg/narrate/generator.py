"""Narrative generator — produces markdown from knowledge graphs."""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import networkx as nx
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from sift_kg.extract.llm_client import LLMClient
from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.narrate.prompts import (
    build_entity_description_prompt,
    build_narrative_prompt,
    build_relationship_chain_prompt,
    build_theme_naming_prompt,
    build_timeline_prompt,
)

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 4
MAX_OVERVIEW_ENTITIES = 50
MAX_OVERVIEW_RELATIONS = 150
MAX_DESCRIBED_ENTITIES = 250

_YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})\b")
_FULL_DATE_PATTERN = re.compile(
    r"\b(\w+ \d{1,2},? (?:19|20)\d{2}"
    r"|\d{1,2}/\d{1,2}/(?:19|20)\d{2}"
    r"|(?:19|20)\d{2}-\d{2}-\d{2})\b"
)


def generate_narrative(
    kg: KnowledgeGraph,
    llm: LLMClient,
    output_dir: Path,
    system_context: str = "",
    include_entity_descriptions: bool = True,
    max_cost: float | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Path:
    """Generate narrative markdown from knowledge graph.

    Returns:
        Path to written narrative.md
    """
    # Gather graph data
    entities = []
    doc_count = 0
    for nid, data in kg.graph.nodes(data=True):
        if data.get("entity_type") == "DOCUMENT":
            if nid.startswith("doc:"):
                doc_count += 1
            continue
        entities.append({
            "id": nid,
            "name": data.get("name", nid),
            "entity_type": data.get("entity_type", "UNKNOWN"),
            "attributes": data.get("attributes", {}),
            "source_documents": data.get("source_documents", []),
        })

    relations = []
    seen_rels = set()
    for source, target, _key, data in kg.graph.edges(data=True, keys=True):
        rel_type = data.get("relation_type", "")
        if rel_type == "MENTIONED_IN":
            continue
        rel_key = (source, target, rel_type)
        if rel_key in seen_rels:
            continue
        seen_rels.add(rel_key)

        source_data = kg.graph.nodes.get(source, {})
        target_data = kg.graph.nodes.get(target, {})
        relations.append({
            "source_name": source_data.get("name", source),
            "target_name": target_data.get("name", target),
            "relation_type": rel_type,
            "evidence": data.get("evidence", ""),
            "_source_id": source,
            "_target_id": target,
        })

    if not entities:
        logger.warning("No entities found in graph")
        return _write_empty_narrative(output_dir)

    # Rank entities by graph degree (most connected first)
    degree_map = dict(kg.graph.degree())
    entities.sort(key=lambda e: degree_map.get(e["id"], 0), reverse=True)
    top_entity_ids = {e["id"] for e in entities[:MAX_OVERVIEW_ENTITIES]}
    top_relations = [
        r for r in relations
        if r.get("_source_id") in top_entity_ids or r.get("_target_id") in top_entity_ids
    ][:MAX_OVERVIEW_RELATIONS]

    # Load source context quotes early — needed for overview + descriptions
    entity_contexts = _load_entity_contexts(output_dir / "extractions")

    logger.info(
        f"Generating narrative: {len(entities)} entities ({min(len(entities), MAX_OVERVIEW_ENTITIES)} in overview), "
        f"{len(relations)} relations ({len(top_relations)} in overview)"
    )

    # --- Phase 1: Overview with journalist tone + source excerpts ---
    prompt = build_narrative_prompt(
        entities[:MAX_OVERVIEW_ENTITIES], top_relations, doc_count, system_context,
        total_entities=len(entities), total_relations=len(relations),
        entity_contexts=entity_contexts,
    )
    try:
        overview = llm.call(prompt)
    except RuntimeError as e:
        logger.error(f"Narrative generation failed: {e}")
        overview = "Narrative generation failed. Please try again."

    # --- Phase 2: Relationship chain narratives ---
    relationship_narrative = ""
    chains = _find_relationship_chains(kg, entities, degree_map)
    if chains:
        chain_prompt = build_relationship_chain_prompt(
            chains, entity_contexts, system_context
        )
        try:
            relationship_narrative = llm.call(chain_prompt)
        except RuntimeError as e:
            logger.warning(f"Relationship chain generation failed: {e}")

    # --- Phase 4: Timeline extraction + narrative ---
    timeline_narrative = ""
    timeline_events = _extract_timeline_events(entities, relations)
    if len(timeline_events) >= 3:
        timeline_prompt = build_timeline_prompt(timeline_events, system_context)
        try:
            timeline_narrative = llm.call(timeline_prompt)
        except RuntimeError as e:
            logger.warning(f"Timeline generation failed: {e}")

    # --- Generate per-entity descriptions (top entities by degree) ---
    entity_descriptions: dict[str, str] = {}
    if include_entity_descriptions and entities:
        described = entities[:MAX_DESCRIBED_ENTITIES]
        skipped = len(entities) - len(described)
        if skipped > 0:
            logger.info(f"Describing top {len(described)} entities, skipping {skipped} low-connectivity entities")

        entity_descriptions = asyncio.run(
            _agenerate_entity_descriptions(
                described, kg, llm, max_cost, concurrency, entity_contexts, system_context
            )
        )

    # --- Phase 3: Community detection + theme naming ---
    communities: list[list[dict]] | None = None
    community_labels: dict[int, str] = {}
    if entity_descriptions:
        communities = _detect_communities(kg, entities, entity_descriptions, degree_map)
        if communities:
            community_labels = asyncio.run(
                _agenerate_theme_labels(communities, kg, llm, concurrency)
            )

    # Save community assignments for visualizer
    if communities and community_labels:
        comm_data: dict[str, str] = {}
        for i, community in enumerate(communities):
            label = community_labels.get(i, f"Community {i + 1}")
            for e in community:
                comm_data[e["id"]] = label
        (output_dir / "communities.json").write_text(
            json.dumps(comm_data, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info(f"Community assignments saved ({len(set(comm_data.values()))} communities)")

    # Save descriptions as JSON sidecar for viewer integration
    if entity_descriptions:
        desc_path = output_dir / "entity_descriptions.json"
        desc_path.write_text(
            json.dumps(entity_descriptions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Descriptions saved: {desc_path}")

    # Build markdown
    md = _build_markdown(
        overview, entities, relations, entity_descriptions, doc_count,
        degree_map=degree_map,
        relationship_narrative=relationship_narrative,
        timeline_narrative=timeline_narrative,
        communities=communities,
        community_labels=community_labels,
    )

    # Write
    output_path = output_dir / "narrative.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")

    logger.info(
        f"Narrative written: {output_path} "
        f"({len(entity_descriptions)} entity descriptions, "
        f"${llm.total_cost_usd:.4f} total cost)"
    )
    return output_path


# ---------------------------------------------------------------------------
# Phase 2 helpers: Relationship chains
# ---------------------------------------------------------------------------


def _find_relationship_chains(
    kg: KnowledgeGraph,
    entities: list[dict[str, Any]],
    degree_map: dict[str, int],
) -> list[list[dict[str, Any]]]:
    """Find interesting relationship paths between top entities.

    Returns list of chains, each being a list of entity dicts along the path.
    Filters to paths of length 2-4 edges (1-3 intermediaries).
    """
    top_ids = [e["id"] for e in entities[:10]]
    if len(top_ids) < 2:
        return []

    try:
        undirected = kg.graph.to_undirected()
    except Exception:
        return []

    # Find shortest paths between pairs of top entities
    candidate_chains: list[tuple[list[str], int]] = []
    for i, src_id in enumerate(top_ids):
        for tgt_id in top_ids[i + 1 :]:
            try:
                path = nx.shortest_path(undirected, src_id, tgt_id)
            except nx.NetworkXNoPath:
                continue

            path_len = len(path) - 1  # number of edges
            if 2 <= path_len <= 4:
                # Score by total degree of intermediaries
                intermediary_degree = sum(
                    degree_map.get(nid, 0) for nid in path[1:-1]
                )
                candidate_chains.append((path, intermediary_degree))

    # Sort by intermediary degree (most interesting first)
    candidate_chains.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate: skip chains that share >50% of nodes with already-selected
    selected: list[list[str]] = []
    selected_nodes: set[str] = set()
    for path, _ in candidate_chains:
        path_set = set(path)
        overlap = len(path_set & selected_nodes)
        if overlap <= len(path) * 0.5:
            selected.append(path)
            selected_nodes.update(path_set)
        if len(selected) >= 6:
            break

    # Convert node IDs to entity dicts
    entity_map = {e["id"]: e for e in entities}
    result: list[list[dict[str, Any]]] = []
    for path in selected:
        chain: list[dict[str, Any]] = []
        for nid in path:
            if nid in entity_map:
                chain.append(entity_map[nid])
            else:
                node_data = kg.graph.nodes.get(nid, {})
                chain.append({
                    "id": nid,
                    "name": node_data.get("name", nid),
                    "entity_type": node_data.get("entity_type", "UNKNOWN"),
                })
        result.append(chain)

    return result


# ---------------------------------------------------------------------------
# Phase 3 helpers: Community detection + theme naming
# ---------------------------------------------------------------------------


def _detect_communities(
    kg: KnowledgeGraph,
    entities: list[dict[str, Any]],
    entity_descriptions: dict[str, str],
    degree_map: dict[str, int],
) -> list[list[dict[str, Any]]] | None:
    """Detect thematic communities using Louvain method.

    Returns list of communities (each a list of entity dicts) sorted by
    total degree. Only includes communities with 3+ described entities.
    Returns None if detection fails or produces <=1 community.
    """
    try:
        undirected = kg.graph.to_undirected()
        raw_communities = nx.community.louvain_communities(undirected)
    except Exception as e:
        logger.debug(f"Community detection failed: {e}")
        return None

    if len(raw_communities) <= 1:
        return None

    described_ids = set(entity_descriptions.keys())
    entity_map = {e["id"]: e for e in entities}

    result: list[list[dict[str, Any]]] = []
    for community_nodes in raw_communities:
        members = [
            entity_map[nid]
            for nid in community_nodes
            if nid in entity_map and nid in described_ids
        ]
        if len(members) >= 3:
            result.append(members)

    if not result:
        return None

    # Sort communities by total degree of members (most connected first)
    result.sort(
        key=lambda c: sum(degree_map.get(e["id"], 0) for e in c),
        reverse=True,
    )

    return result


async def _agenerate_theme_labels(
    communities: list[list[dict[str, Any]]],
    kg: KnowledgeGraph,
    llm: LLMClient,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[int, str]:
    """Generate thematic labels for entity communities via async LLM calls."""
    sem = asyncio.Semaphore(concurrency)

    async def _label(idx: int, community: list[dict[str, Any]]) -> tuple[int, str]:
        names = [e["name"] for e in community]
        types = [e["entity_type"] for e in community]

        # Collect relation types within this community
        community_ids = {e["id"] for e in community}
        rel_types: set[str] = set()
        for eid in community_ids:
            for r in kg.get_relations(eid):
                if r["source"] in community_ids or r["target"] in community_ids:
                    rt = r.get("relation_type", "")
                    if rt and rt != "MENTIONED_IN":
                        rel_types.add(rt)

        prompt = build_theme_naming_prompt(names, types, list(rel_types))
        async with sem:
            try:
                label = await llm.acall(prompt)
                # Strip quotes/whitespace the LLM might add
                return idx, label.strip().strip('"').strip("'")
            except RuntimeError:
                return idx, f"Group {idx + 1}"

    results = await asyncio.gather(*[_label(i, c) for i, c in enumerate(communities)])
    return dict(results)


# ---------------------------------------------------------------------------
# Phase 4 helpers: Timeline extraction
# ---------------------------------------------------------------------------


def _extract_timeline_events(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> list[tuple[str, str, str]]:
    """Extract dated events from entities and relations.

    Returns sorted list of (date_str, entity_name, event_description) tuples.
    """
    events: list[tuple[str, str, str]] = []

    for e in entities:
        attrs = e.get("attributes", {})
        name = e.get("name", "?")

        # Check common date-like attributes
        for attr_key in ("date", "event_date", "year", "start_date", "end_date"):
            val = attrs.get(attr_key, "")
            if val:
                events.append((str(val), name, f"{e.get('entity_type', '')}: {name}"))
                break
        else:
            # EVENT-type entities: scan all attributes for date patterns
            if e.get("entity_type") == "EVENT":
                for val in attrs.values():
                    if isinstance(val, str) and _YEAR_PATTERN.search(val):
                        events.append((val, name, name))
                        break

    # Scan relation evidence for date mentions
    for r in relations:
        evidence = r.get("evidence", "")
        if not isinstance(evidence, str) or not evidence:
            continue
        date_match = _FULL_DATE_PATTERN.search(evidence)
        if not date_match:
            date_match = _YEAR_PATTERN.search(evidence)
        if date_match:
            date_str = date_match.group(1)
            src = r.get("source_name", "?")
            tgt = r.get("target_name", "?")
            rel = r.get("relation_type", "?")
            desc = f"{src} {rel.lower().replace('_', ' ')} {tgt}"
            events.append((date_str, src, desc))

    # Deduplicate and sort chronologically
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str, str]] = []
    for date, name, desc in events:
        key = (date, name)
        if key not in seen:
            seen.add(key)
            unique.append((date, name, desc))

    unique.sort(key=lambda x: x[0])
    return unique


# ---------------------------------------------------------------------------
# Source context loading
# ---------------------------------------------------------------------------


def _load_entity_contexts(extractions_dir: Path) -> dict[str, list[str]]:
    """Load source text context quotes from extraction JSONs.

    Returns a map of entity name (lowercased) -> list of context quotes.
    Handles multi-context fields joined with ' ||| ' separator.
    """
    contexts: dict[str, list[str]] = {}
    if not extractions_dir.exists():
        return contexts

    for f in extractions_dir.glob("*.json"):
        data = json.loads(f.read_text())
        for e in data.get("entities", []):
            ctx = e.get("context", "").strip()
            if ctx:
                key = e.get("name", "").lower().strip()
                # Split multi-context fields (from dedup merging)
                for fragment in ctx.split(" ||| "):
                    fragment = fragment.strip()
                    if fragment:
                        contexts.setdefault(key, []).append(fragment)

    return contexts


# ---------------------------------------------------------------------------
# Entity description generation
# ---------------------------------------------------------------------------


async def _agenerate_entity_descriptions(
    entities: list[dict],
    kg: KnowledgeGraph,
    llm: LLMClient,
    max_cost: float | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    entity_contexts: dict[str, list[str]] | None = None,
    system_context: str = "",
) -> dict[str, str]:
    """Generate descriptions for each entity with async concurrency."""
    sem = asyncio.Semaphore(concurrency)
    completed = 0
    entity_contexts = entity_contexts or {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("${task.fields[cost]:.3f}"),
    ) as progress:
        ptask = progress.add_task("Entity descriptions...", total=len(entities), cost=0.0)

        async def _describe(entity: dict) -> tuple[str, str | None]:
            nonlocal completed
            if max_cost and llm.total_cost_usd >= max_cost:
                return entity["id"], None

            eid = entity["id"]
            rels = kg.get_relations(eid)
            rel_dicts = []
            for r in rels:
                source_data = kg.graph.nodes.get(r["source"], {})
                target_data = kg.graph.nodes.get(r["target"], {})
                rel_dicts.append({
                    "source_name": source_data.get("name", r["source"]),
                    "target_name": target_data.get("name", r["target"]),
                    "relation_type": r.get("relation_type", ""),
                })

            # Get source text quotes for this entity
            source_quotes = entity_contexts.get(entity["name"].lower().strip(), [])

            prompt = build_entity_description_prompt(
                entity_name=entity["name"],
                entity_type=entity["entity_type"],
                attributes=entity.get("attributes", {}),
                relations=rel_dicts,
                source_documents=entity.get("source_documents", []),
                source_contexts=source_quotes,
                system_context=system_context,
            )

            async with sem:
                try:
                    desc = await llm.acall(prompt)
                    completed += 1
                    progress.update(ptask, completed=completed, cost=llm.total_cost_usd)
                    return eid, desc.strip()
                except RuntimeError as e:
                    logger.warning(f"Failed to generate description for {entity['name']}: {e}")
                    completed += 1
                    progress.update(ptask, completed=completed, cost=llm.total_cost_usd)
                    return eid, None

        results = await asyncio.gather(*[_describe(e) for e in entities])

    if max_cost and llm.total_cost_usd >= max_cost:
        logger.warning("Budget limit reached during entity descriptions")

    return {eid: desc for eid, desc in results if desc is not None}


# ---------------------------------------------------------------------------
# Markdown assembly
# ---------------------------------------------------------------------------


def _build_markdown(
    overview: str,
    entities: list[dict],
    relations: list[dict],
    entity_descriptions: dict[str, str],
    doc_count: int,
    degree_map: dict[str, int] | None = None,
    relationship_narrative: str = "",
    timeline_narrative: str = "",
    communities: list[list[dict]] | None = None,
    community_labels: dict[int, str] | None = None,
) -> str:
    """Build the narrative markdown document."""
    degree = degree_map or {}

    lines = [
        "# Knowledge Graph Narrative",
        "",
        f"*Generated from {doc_count} documents, "
        f"{len(entities)} entities, {len(relations)} relations.*",
        "",
        "---",
        "",
        "## Overview",
        "",
        overview.strip(),
        "",
    ]

    # Key Connections section (Phase 2)
    if relationship_narrative:
        lines.extend([
            "---",
            "",
            "## Key Connections",
            "",
            relationship_narrative.strip(),
            "",
        ])

    # Timeline section (Phase 4)
    if timeline_narrative:
        lines.extend([
            "---",
            "",
            "## Timeline",
            "",
            timeline_narrative.strip(),
            "",
        ])

    # Entity Profiles section
    if entity_descriptions:
        lines.extend([
            "---",
            "",
            "## Entity Profiles",
            "",
        ])

        if communities and community_labels:
            # Phase 3: Thematic grouping
            assigned_ids: set[str] = set()
            for i, community in enumerate(communities):
                label = community_labels.get(i, f"Group {i + 1}")
                lines.append(f"### {label}")
                lines.append("")
                # Sort by degree within community
                sorted_members = sorted(
                    community,
                    key=lambda e: degree.get(e["id"], 0),
                    reverse=True,
                )
                for e in sorted_members:
                    desc = entity_descriptions.get(e["id"], "")
                    if desc:
                        lines.append(f"**{e['name']}**: {desc}")
                        lines.append("")
                    assigned_ids.add(e["id"])

            # Entities not in any significant community
            others = [
                e for e in entities
                if e["id"] not in assigned_ids and e["id"] in entity_descriptions
            ]
            if others:
                lines.append("### Other Entities")
                lines.append("")
                others.sort(key=lambda e: degree.get(e["id"], 0), reverse=True)
                for e in others:
                    desc = entity_descriptions.get(e["id"], "")
                    if desc:
                        lines.append(f"**{e['name']}**: {desc}")
                        lines.append("")
        else:
            # Fallback: type-based grouping, sorted by degree (Phase 1)
            type_groups: dict[str, list[dict]] = {}
            for e in entities:
                type_groups.setdefault(e["entity_type"], []).append(e)

            for etype in sorted(type_groups.keys()):
                lines.append(f"### {etype}")
                lines.append("")
                for e in sorted(
                    type_groups[etype],
                    key=lambda x: degree.get(x["id"], 0),
                    reverse=True,
                ):
                    desc = entity_descriptions.get(e["id"], "")
                    if desc:
                        lines.append(f"**{e['name']}**: {desc}")
                        lines.append("")

    lines.extend([
        "---",
        "",
        "*This narrative was generated by sift-kg using AI analysis. "
        "All content should be verified against source documents.*",
    ])

    return "\n".join(lines)


def _write_empty_narrative(output_dir: Path) -> Path:
    """Write a placeholder when no entities exist."""
    output_path = output_dir / "narrative.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# Knowledge Graph Narrative\n\nNo entities found in the graph.\n"
    )
    return output_path
