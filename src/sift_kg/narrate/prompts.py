"""Narrative generation prompts — domain-agnostic."""

import json
from typing import Any


def build_narrative_prompt(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    document_count: int,
    system_context: str = "",
) -> str:
    """Build prompt for generating overview narrative from graph data.

    Args:
        entities: Non-document entities with name, type, attributes
        relations: Relations with source_name, target_name, relation_type
        document_count: Total documents processed
        system_context: Optional domain context from domain config
    """
    # Summarize entities by type
    type_groups: dict[str, list[str]] = {}
    for e in entities:
        etype = e.get("entity_type", "UNKNOWN")
        type_groups.setdefault(etype, []).append(e.get("name", "?"))

    entity_summary = ""
    for etype, names in sorted(type_groups.items()):
        entity_summary += f"\n{etype} ({len(names)}): {', '.join(names[:20])}"
        if len(names) > 20:
            entity_summary += f" ... and {len(names) - 20} more"

    # Summarize relations
    rel_lines = []
    for r in relations[:50]:  # Cap to avoid huge prompts
        rel_lines.append(
            f"- {r.get('source_name', '?')} --[{r.get('relation_type', '?')}]--> "
            f"{r.get('target_name', '?')}"
        )
    if len(relations) > 50:
        rel_lines.append(f"... and {len(relations) - 50} more relations")

    relations_text = "\n".join(rel_lines) if rel_lines else "No relations found."

    context_section = ""
    if system_context:
        context_section = f"\nDOMAIN CONTEXT:\n{system_context[:2000]}\n"

    return f"""{context_section}You are analyzing a knowledge graph extracted from {document_count} documents.

ENTITIES:{entity_summary}

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
) -> str:
    """Build prompt for generating a description of a single entity.

    Args:
        entity_name: Name of the entity
        entity_type: Type (PERSON, ORGANIZATION, etc.)
        attributes: Entity attributes from extraction
        relations: Relations involving this entity
        source_documents: Documents mentioning this entity
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

    return f"""Write a brief description (2-4 sentences) of this entity based on the knowledge graph data.

ENTITY: {entity_name}
TYPE: {entity_type}
ATTRIBUTES: {attrs_text}

RELATIONS:
{relations_text}

MENTIONED IN DOCUMENTS: {docs_text}

Describe who/what this entity is and their significance based on the available data.
Be specific and factual. Do not speculate beyond what the data shows.

Output ONLY the description. No headers or formatting."""
