"""Narrative generation prompts — domain-agnostic."""

import json
from typing import Any


def build_narrative_prompt(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    document_count: int,
    system_context: str = "",
    total_entities: int | None = None,
    total_relations: int | None = None,
) -> str:
    """Build prompt for generating overview narrative from graph data.

    Entities should be pre-ranked by importance (e.g. graph degree).
    The prompt receives only the top entities and their relations.
    """
    total_ent = total_entities or len(entities)
    total_rel = total_relations or len(relations)

    # Summarize entities by type
    type_groups: dict[str, list[str]] = {}
    for e in entities:
        etype = e.get("entity_type", "UNKNOWN")
        type_groups.setdefault(etype, []).append(e.get("name", "?"))

    entity_summary = ""
    for etype, names in sorted(type_groups.items()):
        entity_summary += f"\n{etype} ({len(names)}): {', '.join(names)}"

    # Format relations
    rel_lines = []
    for r in relations:
        rel_lines.append(
            f"- {r.get('source_name', '?')} --[{r.get('relation_type', '?')}]--> "
            f"{r.get('target_name', '?')}"
        )
    relations_text = "\n".join(rel_lines) if rel_lines else "No relations found."

    scope_note = ""
    if total_ent > len(entities):
        scope_note = (
            f"\nNote: Showing the {len(entities)} most connected entities out of "
            f"{total_ent} total, and {len(relations)} of {total_rel} relations.\n"
        )

    context_section = ""
    if system_context:
        context_section = f"\nDOMAIN CONTEXT:\n{system_context[:2000]}\n"

    return f"""{context_section}You are analyzing a knowledge graph extracted from {document_count} documents ({total_ent} entities, {total_rel} relations total).
{scope_note}
TOP ENTITIES (ranked by connectivity):{entity_summary}

KEY RELATIONS:
{relations_text}

Write a narrative summary (3-5 paragraphs) that:
1. Describes the key people, organizations, and locations found
2. Explains the major relationships and connections between entities
3. Highlights any notable patterns or clusters of activity
4. Notes what the documents collectively reveal

Write in an engaging, analytical style. Be specific — name entities and describe
their connections. This is a research summary, not a legal document.

Output ONLY the narrative prose. No headers, no bullet points, no metadata."""


def build_entity_description_prompt(
    entity_name: str,
    entity_type: str,
    attributes: dict[str, Any],
    relations: list[dict[str, Any]],
    source_documents: list[str],
    source_contexts: list[str] | None = None,
) -> str:
    """Build prompt for generating a description of a single entity.

    Args:
        entity_name: Name of the entity
        entity_type: Type (PERSON, ORGANIZATION, etc.)
        attributes: Entity attributes from extraction
        relations: Relations involving this entity
        source_documents: Documents mentioning this entity
        source_contexts: Quotes from source text where entity appears
    """
    attrs_text = json.dumps(attributes, indent=2, ensure_ascii=False) if attributes else "None"

    rel_lines = []
    for r in relations[:20]:
        rel_lines.append(
            f"- {r.get('relation_type', '?')}: "
            f"{r.get('source_name', '?')} → {r.get('target_name', '?')}"
        )
    relations_text = "\n".join(rel_lines) if rel_lines else "No known relations."

    docs_text = ", ".join(source_documents[:10]) if source_documents else "Unknown"

    # Deduplicate and cap source contexts
    contexts = []
    if source_contexts:
        seen = set()
        for ctx in source_contexts:
            normalized = ctx.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                contexts.append(ctx)
    contexts = contexts[:15]  # Cap to avoid oversized prompts
    contexts_text = "\n".join(f"- \"{ctx}\"" for ctx in contexts) if contexts else "None available."

    return f"""Write a brief description (2-4 sentences) of this entity based on the source documents and knowledge graph data.

ENTITY: {entity_name}
TYPE: {entity_type}
ATTRIBUTES: {attrs_text}

SOURCE TEXT EXCERPTS:
{contexts_text}

RELATIONS:
{relations_text}

MENTIONED IN DOCUMENTS: {docs_text}

Synthesize the source excerpts and relations into a specific, factual description.
Focus on what the documents actually say about this entity — their role, actions, and significance.
Do not provide generic background information that isn't in the sources.

Output ONLY the description. No headers or formatting."""
