"""Narrative generator — produces markdown from knowledge graphs."""

import logging
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

from sift_kg.extract.llm_client import LLMClient
from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.narrate.prompts import build_entity_description_prompt, build_narrative_prompt

logger = logging.getLogger(__name__)


def generate_narrative(
    kg: KnowledgeGraph,
    llm: LLMClient,
    output_dir: Path,
    system_context: str = "",
    include_entity_descriptions: bool = True,
    max_cost: float | None = None,
) -> Path:
    """Generate narrative markdown from knowledge graph.

    Args:
        kg: Populated knowledge graph
        llm: LLM client
        output_dir: Where to write narrative.md
        system_context: Optional domain context
        include_entity_descriptions: Whether to generate per-entity descriptions
        max_cost: Budget limit for narrative generation

    Returns:
        Path to written narrative.md
    """
    # Gather graph data
    entities = []
    doc_count = 0
    for nid, data in kg.graph.nodes(data=True):
        if data.get("entity_type") == "DOCUMENT":
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
        })

    if not entities:
        logger.warning("No entities found in graph")
        return _write_empty_narrative(output_dir)

    # Generate overview narrative
    logger.info(f"Generating narrative: {len(entities)} entities, {len(relations)} relations")

    prompt = build_narrative_prompt(entities, relations, doc_count, system_context)
    try:
        overview = llm.call(prompt)
    except RuntimeError as e:
        logger.error(f"Narrative generation failed: {e}")
        overview = "Narrative generation failed. Please try again."

    # Generate per-entity descriptions
    entity_descriptions: dict[str, str] = {}
    if include_entity_descriptions and entities:
        entity_descriptions = _generate_entity_descriptions(
            entities, kg, llm, max_cost
        )

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


def _generate_entity_descriptions(
    entities: list[dict],
    kg: KnowledgeGraph,
    llm: LLMClient,
    max_cost: float | None = None,
) -> dict[str, str]:
    """Generate descriptions for each entity."""
    descriptions: dict[str, str] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        task = progress.add_task("Entity descriptions...", total=len(entities))

        for entity in entities:
            if max_cost and llm.total_cost_usd >= max_cost:
                logger.warning("Budget limit reached during entity descriptions")
                break

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

            prompt = build_entity_description_prompt(
                entity_name=entity["name"],
                entity_type=entity["entity_type"],
                attributes=entity.get("attributes", {}),
                relations=rel_dicts,
                source_documents=entity.get("source_documents", []),
            )

            try:
                desc = llm.call(prompt)
                descriptions[eid] = desc.strip()
            except RuntimeError as e:
                logger.warning(f"Failed to generate description for {entity['name']}: {e}")

            progress.advance(task)

    return descriptions


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
