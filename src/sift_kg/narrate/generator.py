"""Narrative generator — produces markdown from knowledge graphs."""

import asyncio
import json
import logging
from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from sift_kg.extract.llm_client import LLMClient
from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.narrate.prompts import build_entity_description_prompt, build_narrative_prompt

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 4
MAX_OVERVIEW_ENTITIES = 50
MAX_OVERVIEW_RELATIONS = 150
MAX_DESCRIBED_ENTITIES = 100


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
            "_source_id": source,
            "_target_id": target,
        })

    if not entities:
        logger.warning("No entities found in graph")
        return _write_empty_narrative(output_dir)

    # Rank entities by graph degree (most connected first) for the overview.
    degree_map = dict(kg.graph.degree())
    entities.sort(key=lambda e: degree_map.get(e["id"], 0), reverse=True)
    top_entity_ids = {e["id"] for e in entities[:MAX_OVERVIEW_ENTITIES]}
    top_relations = [
        r for r in relations
        if r.get("_source_id") in top_entity_ids or r.get("_target_id") in top_entity_ids
    ][:MAX_OVERVIEW_RELATIONS]

    logger.info(
        f"Generating narrative: {len(entities)} entities ({min(len(entities), MAX_OVERVIEW_ENTITIES)} in overview), "
        f"{len(relations)} relations ({len(top_relations)} in overview)"
    )

    prompt = build_narrative_prompt(
        entities[:MAX_OVERVIEW_ENTITIES], top_relations, doc_count, system_context,
        total_entities=len(entities), total_relations=len(relations),
    )
    try:
        overview = llm.call(prompt)
    except RuntimeError as e:
        logger.error(f"Narrative generation failed: {e}")
        overview = "Narrative generation failed. Please try again."

    # Generate per-entity descriptions (top entities by degree only)
    entity_descriptions: dict[str, str] = {}
    if include_entity_descriptions and entities:
        described = entities[:MAX_DESCRIBED_ENTITIES]
        skipped = len(entities) - len(described)
        if skipped > 0:
            logger.info(f"Describing top {len(described)} entities, skipping {skipped} low-connectivity entities")

        # Load source context quotes from extractions
        entity_contexts = _load_entity_contexts(output_dir / "extractions")

        entity_descriptions = asyncio.run(
            _agenerate_entity_descriptions(
                described, kg, llm, max_cost, concurrency, entity_contexts
            )
        )

    # Save descriptions as JSON sidecar for viewer integration
    if entity_descriptions:
        desc_path = output_dir / "entity_descriptions.json"
        desc_path.write_text(
            json.dumps(entity_descriptions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Descriptions saved: {desc_path}")

    # Build markdown
    md = _build_markdown(overview, entities, relations, entity_descriptions, doc_count)

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


def _load_entity_contexts(extractions_dir: Path) -> dict[str, list[str]]:
    """Load source text context quotes from extraction JSONs.

    Returns a map of entity name (lowercased) → list of context quotes.
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
                contexts.setdefault(key, []).append(ctx)

    return contexts


async def _agenerate_entity_descriptions(
    entities: list[dict],
    kg: KnowledgeGraph,
    llm: LLMClient,
    max_cost: float | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    entity_contexts: dict[str, list[str]] | None = None,
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


def _build_markdown(
    overview: str,
    entities: list[dict],
    relations: list[dict],
    entity_descriptions: dict[str, str],
    doc_count: int,
) -> str:
    """Build the narrative markdown document."""
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

    if entity_descriptions:
        lines.extend([
            "---",
            "",
            "## Entity Profiles",
            "",
        ])

        # Group by type
        type_groups: dict[str, list[dict]] = {}
        for e in entities:
            type_groups.setdefault(e["entity_type"], []).append(e)

        for etype in sorted(type_groups.keys()):
            lines.append(f"### {etype}")
            lines.append("")
            for e in sorted(type_groups[etype], key=lambda x: x["name"]):
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
