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
    entity_contexts: dict[str, list[str]] | None = None,
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

    # Collect top source excerpts from entity contexts
    source_excerpts_section = ""
    if entity_contexts:
        excerpts: list[str] = []
        for e in entities[:20]:
            name_key = e.get("name", "").lower().strip()
            quotes = entity_contexts.get(name_key, [])
            for q in quotes[:3]:
                if len(q) > 40:
                    excerpts.append(q)
                if len(excerpts) >= 25:
                    break
            if len(excerpts) >= 25:
                break
        if excerpts:
            formatted = "\n".join(f'- "{ex}"' for ex in excerpts)
            source_excerpts_section = (
                f"\nSOURCE EXCERPTS (verbatim from documents):\n{formatted}\n"
            )

    return f"""{context_section}You are analyzing a knowledge graph extracted from {document_count} documents ({total_ent} entities, {total_rel} relations total).
{scope_note}
TOP ENTITIES (ranked by connectivity):{entity_summary}

KEY RELATIONS:
{relations_text}
{source_excerpts_section}
Write an overview (3-5 paragraphs) like an investigative journalist briefing an editor. Be direct, specific, and assertive.

Rules:
1. Name names. Describe what people did, who they worked with, where things happened.
2. Trace the key connections — who links to whom and through what.
3. Identify clusters of activity and what they reveal.
4. Use the source excerpts for specific details, quotes, and evidence.
5. NEVER use: "is associated with", "highlighting", "underscoring", "indicating", "in the context of", "is significant as", "this suggests."
6. NEVER reference the documents themselves — no "the documents reveal", "records show", "as evidenced by."
7. State facts. Be specific. No hedging, no filler.

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
    for r in relations[:40]:
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
    contexts = contexts[:30]
    contexts_text = "\n".join(f'- "{ctx}"' for ctx in contexts) if contexts else "None available."

    context_section = ""
    if system_context:
        context_section = f"DOMAIN CONTEXT:\n{system_context}\n\n"

    return f"""{context_section}Write an investigative profile of {entity_name} ({entity_type}) based on the evidence below.

SOURCE TEXT EXCERPTS:
{contexts_text}

ATTRIBUTES: {attrs_text}

CONNECTIONS:
{relations_text}

Length: Proportional to evidence. Rich source material = multiple paragraphs. Thin evidence = 2-3 sentences. Never pad.

OPENING SENTENCE — THIS IS CRITICAL:
Your first sentence must be a concrete action or fact. Never a role description.
- Bad: "Sarah Kellen played a significant role in Epstein's network." → This says NOTHING.
- Good: "Sarah Kellen scheduled appointments, answered phones, and brought girls to Epstein's Palm Beach mansion."
- Bad: "Ghislaine Maxwell played a pivotal role in the sex trafficking operation." → Vague filler.
- Good: "Ghislaine Maxwell recruited underage girls for Epstein, trained them in massage techniques, and traveled with them internationally."
- Bad: "Laura Menninger is a key legal figure in the case." → Wikipedia summary voice.
- Good: "Laura Menninger represented Ghislaine Maxwell in the Giuffre v. Maxwell defamation suit, filing motions to seal depositions and challenging witness credibility."
- Bad: "The FBI investigated Epstein." → Too vague.
- Good: "The FBI opened an investigation into Epstein in 2006 after Palm Beach police referred the case, interviewing over 30 victims."

STYLE: Every sentence must contain a SPECIFIC VERB describing what someone DID, SAID, WENT, FILED, RECRUITED, TESTIFIED, PAID, FLEW, etc. If you catch yourself writing "was involved in" or "was connected to" — replace it with the actual action.

BANNED PATTERNS (instant fail):
- "[Name] played a [significant/pivotal/key/central/crucial/important] role" — NEVER. Say what they DID.
- "[Name] is a [key/significant/important/notable] [figure/person/entity]" — NEVER. Say what they DID.
- "[Name] served as" — replace with active verbs (represented, defended, managed, ran)
- "as evidenced by", "the documents reveal", "court records show", "the source material indicates"
- "is associated with", "is mentioned in", "in the context of", "highlighting", "indicating", "underscoring"
- "is significant as", "is noted for", "is identified as", "this suggests", "this indicates"
- "also known as" in the opening — weave aliases in naturally or skip them
- Any sentence where you could swap the person's name and it would still be true. That means it's generic.

RULES:
- ONLY use facts from the source evidence. No outside knowledge.
- Lead with their most important ACTION, not a label or role description.
- Include specifics: dates, locations, quotes, names of people they interacted with.
- State facts. Active voice. Subject-verb-object.

Output ONLY the profile text."""


def build_relationship_chain_prompt(
    chains: list[list[dict[str, Any]]],
    entity_contexts: dict[str, list[str]] | None = None,
    system_context: str = "",
) -> str:
    """Build prompt for narrating relationship chains between key entities.

    Each chain is a list of entity dicts representing a path through the graph.
    """
    context_section = ""
    if system_context:
        context_section = f"DOMAIN CONTEXT:\n{system_context[:1000]}\n\n"

    chain_descriptions = []
    all_entity_names: set[str] = set()
    for i, chain in enumerate(chains, 1):
        names = [e["name"] for e in chain]
        all_entity_names.update(n.lower().strip() for n in names)
        path_str = " \u2192 ".join(names)
        chain_descriptions.append(f"Chain {i}: {path_str}")

    chains_text = "\n".join(chain_descriptions)

    # Include relevant source contexts
    contexts_section = ""
    if entity_contexts:
        relevant_quotes: list[str] = []
        for name in sorted(all_entity_names):
            quotes = entity_contexts.get(name, [])
            for q in quotes[:2]:
                if len(q) > 40:
                    relevant_quotes.append(q)
            if len(relevant_quotes) >= 20:
                break
        if relevant_quotes:
            formatted = "\n".join(f'- "{q}"' for q in relevant_quotes)
            contexts_section = f"\nSOURCE EXCERPTS:\n{formatted}\n"

    return f"""{context_section}Below are relationship chains connecting key entities in this network. Each chain traces a path of connections between important figures.

CHAINS:
{chains_text}
{contexts_section}
Write one paragraph per chain. For each, explain HOW these entities connect — what the intermediaries did, what role they played, why this chain matters.

Rules:
- Write like investigative journalism. Be direct and specific.
- Don't just list the chain — narrate it. What happened along this path?
- Use source excerpts for concrete details where available.
- NEVER use: "is associated with", "highlighting", "underscoring", "indicating", "in the context of."
- NEVER reference documents — no "records show", "documents reveal."
- If a chain's connection is trivial or unclear from the evidence, write a shorter paragraph acknowledging the link without inventing details.

Output ONLY the narrative paragraphs, one per chain. No headers, no bullet points, no chain labels."""


def build_theme_naming_prompt(
    communities: list[dict[str, Any]],
) -> str:
    """Build prompt to generate distinct thematic labels for all communities at once.

    Each community dict has keys: entity_names, entity_types, relation_types.
    """
    cluster_descriptions = []
    for i, comm in enumerate(communities, 1):
        names = ", ".join(comm["entity_names"][:15])
        types = ", ".join(sorted(set(comm["entity_types"])))
        rels = (
            ", ".join(sorted(set(comm["relation_types"]))[:10])
            if comm["relation_types"]
            else "various"
        )
        cluster_descriptions.append(
            f"Cluster {i} ({len(comm['entity_names'])} entities):\n"
            f"  Members: {names}\n"
            f"  Types: {types}\n"
            f"  Relations: {rels}"
        )

    clusters_text = "\n\n".join(cluster_descriptions)

    return f"""Below are {len(communities)} entity clusters from a knowledge graph. Each cluster is a group of closely connected entities.

{clusters_text}

Give each cluster a short, distinctive label (2-6 words) that differentiates it from the others. Each label must be UNIQUE — no two labels should be interchangeable. Focus on WHAT MAKES EACH CLUSTER DIFFERENT, not what they share.

Good labels name specific people, roles, locations, or activities unique to that cluster:
- "Epstein's Inner Circle" vs "Palm Beach Investigation" vs "Giuffre's Legal Team"
Bad labels are generic and interchangeable:
- "Legal Network" vs "Legal Proceedings" vs "Legal Connections"

Return one label per line, in order (Cluster 1 first). No numbering, no quotes, no explanation — just the labels."""


def build_timeline_prompt(
    events: list[tuple[str, str, str]],
    system_context: str = "",
) -> str:
    """Build prompt for generating a narrative timeline.

    Events are (date_str, entity_name, description) tuples, pre-sorted chronologically.
    """
    context_section = ""
    if system_context:
        context_section = f"DOMAIN CONTEXT:\n{system_context[:1000]}\n\n"

    events_text = "\n".join(
        f"- {date}: {name} \u2014 {desc}" for date, name, desc in events
    )

    return f"""{context_section}Below is a chronological list of events extracted from a knowledge graph:

{events_text}

Write a narrative timeline (flowing prose, not bullet points) that connects these events chronologically. Show how earlier events led to later ones. Note acceleration, patterns, or turning points.

Rules:
- Write like investigative journalism — direct, specific, no hedging.
- Include dates and names. Be concrete.
- NEVER use: "highlighting", "underscoring", "indicating", "in the context of."
- Connect events into a story, don't just restate the list.

Output ONLY the narrative timeline."""
