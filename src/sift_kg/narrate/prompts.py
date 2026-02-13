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
    system_context: str = "",
) -> str:
    """Build prompt for generating a description of a single entity.

    Args:
        entity_name: Name of the entity
        entity_type: Type (PERSON, ORGANIZATION, etc.)
        attributes: Entity attributes from extraction
        relations: Relations involving this entity
        source_documents: Documents mentioning this entity
        source_contexts: Quotes from source text where entity appears
        system_context: Domain context for better descriptions
    """
    attrs_text = json.dumps(attributes, indent=2, ensure_ascii=False) if attributes else "None"

    # Format relations as readable sentences, grouping by direction
    outgoing = []
    incoming = []
    for r in relations[:20]:
        rel = r.get("relation_type", "?")
        src = r.get("source_name", "?")
        tgt = r.get("target_name", "?")
        if src == entity_name:
            outgoing.append(f"- {rel} → {tgt}")
        else:
            incoming.append(f"- {src} → {rel}")
    rel_parts = []
    if outgoing:
        rel_parts.extend(outgoing)
    if incoming:
        rel_parts.extend(incoming)
    relations_text = "\n".join(rel_parts) if rel_parts else "No known relations."

    # Deduplicate and cap source contexts
    contexts = []
    if source_contexts:
        seen = set()
        for ctx in source_contexts:
            normalized = ctx.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                contexts.append(ctx)
    contexts = contexts[:15]
    contexts_text = "\n".join(f"- \"{ctx}\"" for ctx in contexts) if contexts else "None available."

    context_section = ""
    if system_context:
        context_section = f"DOMAIN CONTEXT:\n{system_context}\n\n"

    return f"""{context_section}Write a 2-4 sentence description of {entity_name} ({entity_type}).

SOURCE TEXT EXCERPTS:
{contexts_text}

ATTRIBUTES: {attrs_text}

CONNECTIONS:
{relations_text}

Rules:
- ONLY describe what the source documents reveal about this entity. Do NOT add general knowledge, background facts, or geographic descriptions.
- Bad: "Fort Lauderdale is a city known for its boating canals and beaches." Good: "Fort Lauderdale is where law firm Boies Schiller & Flexner LLP operated during the case."
- Describe what this entity DID, not just who they are connected to.
- Lead with their role or significance, then key actions or allegations.
- NEVER mention document filenames, case numbers, or that something was "referenced in" a document.
- NEVER use filler like "is associated with", "is mentioned in", "is referenced in", "in the context of", "highlighting", "indicating".
- State facts directly: "Maxwell recruited girls for Epstein" not "Maxwell is associated with Epstein".
- If the source excerpts are thin, write a shorter description. Do NOT pad with filler.

Output ONLY the description text."""
