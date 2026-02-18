"""LLM extraction prompts — domain-driven, zero-shot.

Provides both separate (entity-only, relation-only) and combined prompts.
The combined prompt halves LLM calls per chunk at near-identical accuracy.
"""

import json

from sift_kg.domains.models import DomainConfig


def build_entity_prompt(
    text: str,
    document_id: str,
    domain: DomainConfig,
) -> str:
    """Build entity extraction prompt from domain config.

    Args:
        text: Document text to extract from
        document_id: Document identifier for context
        domain: Domain configuration with entity types

    Returns:
        Formatted prompt string
    """
    # Build entity type descriptions from domain config
    type_lines = []
    for name, cfg in domain.entity_types.items():
        desc = cfg.description or name
        hints = ""
        if cfg.extraction_hints:
            hints = " (" + "; ".join(cfg.extraction_hints) + ")"
        if cfg.canonical_names:
            names_list = ", ".join(cfg.canonical_names)
            hints += f" ALLOWED VALUES (use ONLY these exact names): [{names_list}]"
        type_lines.append(f"- {name}: {desc}{hints}")

    entity_types_section = "\n".join(type_lines)

    context_section = ""
    if domain.system_context:
        context_section = f"\n{domain.system_context}\n"

    return f"""{context_section}Extract entities from the following document text. Return valid JSON only.

ENTITY TYPES:
{entity_types_section}

OUTPUT SCHEMA:
{{
  "entities": [
    {{
      "name": "string",
      "entity_type": "TYPE_NAME",
      "attributes": {{"key": "value"}},
      "confidence": 0.0-1.0,
      "context": "quote from text where entity appears"
    }}
  ]
}}

RULES:
- Extract ALL entity types present, not just the most common
- Extract only explicit information from the text
- The text may be in ANY language. Extract entities regardless of source language.
- Output all entity names and attribute values in English. Use the most internationally recognized form of each name — do not anglicize personal names (Juan stays Juan, not John; 习近平 → Xi Jinping, not "Xi Near-Peace").
- Keep context quotes in the original language of the source text.
- confidence: 0.0-1.0 based on text clarity
- attributes: include any relevant details (dates, roles, descriptions, etc.)

Document: {document_id}

TEXT:
{text}

OUTPUT JSON:"""


def build_combined_prompt(
    text: str,
    document_id: str,
    domain: DomainConfig,
    doc_context: str = "",
) -> str:
    """Build a single prompt that extracts both entities and relations.

    Cuts LLM calls per chunk from 2 to 1. The prompt instructs the model
    to identify entities first, then find relations between them.

    Args:
        doc_context: Optional document-level summary prepended to each chunk
            so the LLM has context about the overall document.
    """
    type_lines = []
    for name, cfg in domain.entity_types.items():
        desc = cfg.description or name
        hints = ""
        if cfg.extraction_hints:
            hints = " (" + "; ".join(cfg.extraction_hints) + ")"
        if cfg.canonical_names:
            names_list = ", ".join(cfg.canonical_names)
            hints += f" ALLOWED VALUES (use ONLY these exact names): [{names_list}]"
        type_lines.append(f"- {name}: {desc}{hints}")

    entity_types_section = "\n".join(type_lines)
    rel_types = ", ".join(domain.relation_types.keys())

    # Build direction hints for relation types that have source/target constraints
    direction_lines = []
    for rel_name, rel_cfg in domain.relation_types.items():
        if rel_cfg.source_types and rel_cfg.target_types:
            src = "/".join(rel_cfg.source_types)
            tgt = "/".join(rel_cfg.target_types)
            direction_lines.append(f"- {rel_name}: {src} → {tgt}")

    direction_section = ""
    if direction_lines:
        direction_section = (
            "\n\nRELATION DIRECTIONS (source_entity → target_entity):\n"
            + "\n".join(direction_lines)
            + "\nIMPORTANT: source_entity must be the type on the LEFT, target_entity the type on the RIGHT."
        )

    context_section = ""
    if domain.system_context:
        context_section = f"\n{domain.system_context}\n"

    doc_context_section = ""
    if doc_context:
        doc_context_section = (
            "\nDOCUMENT CONTEXT (applies to entire document, not just this excerpt):\n"
            f"{doc_context}\n"
        )

    return f"""{context_section}Extract entities and relationships from the following document text. Return valid JSON only.

STEP 1 — ENTITIES
Identify all entities in the text.

ENTITY TYPES:
{entity_types_section}

STEP 2 — RELATIONS
Identify relationships between the entities you extracted.

RELATION TYPES (use ONLY these — do not invent new types): {rel_types}
If a relationship doesn't fit any listed type, use ASSOCIATED_WITH as the fallback.{direction_section}

OUTPUT SCHEMA:
{{
  "entities": [
    {{
      "name": "string",
      "entity_type": "TYPE_NAME",
      "attributes": {{"key": "value"}},
      "confidence": 0.0-1.0,
      "context": "quote from text where entity appears"
    }}
  ],
  "relations": [
    {{
      "relation_type": "TYPE_NAME",
      "source_entity": "entity name",
      "target_entity": "entity name",
      "confidence": 0.0-1.0,
      "evidence": "quote from text supporting this relation"
    }}
  ]
}}

RULES:
- Extract ALL entity types present, not just the most common
- Extract only explicit information from the text
- The text may be in ANY language. Extract entities regardless of source language.
- Output all entity names and attribute values in English. Use the most internationally recognized form of each name — do not anglicize personal names (Juan stays Juan, not John; 习近平 → Xi Jinping, not "Xi Near-Peace").
- Keep context and evidence quotes in the original language of the source text.
- confidence: 0.0-1.0 based on text clarity
- attributes: include any relevant details (dates, roles, descriptions, etc.)
- Use entity NAMES (not IDs) for source_entity and target_entity
- Only extract explicit relationships stated in the text
- Do not infer relationships from co-occurrence alone
- If no relations found, return an empty relations list

Document: {document_id}
{doc_context_section}
TEXT:
{text}

OUTPUT JSON:"""


def build_relation_prompt(
    text: str,
    entities: list[dict],
    document_id: str,
    domain: DomainConfig,
) -> str:
    """Build relation extraction prompt from domain config.

    Args:
        text: Document text
        entities: Previously extracted entities (list of dicts with name, entity_type)
        document_id: Document identifier
        domain: Domain configuration with relation types

    Returns:
        Formatted prompt string
    """
    # Build relation types list
    rel_types = ", ".join(domain.relation_types.keys())

    entities_json = json.dumps(entities, indent=2, ensure_ascii=False)

    context_section = ""
    if domain.system_context:
        context_section = f"\n{domain.system_context}\n"

    return f"""{context_section}Extract relationships between entities from this document. Return valid JSON only.

RELATION TYPES (use ONLY these — do not invent new types): {rel_types}
If a relationship doesn't fit any listed type, use ASSOCIATED_WITH as the fallback.

OUTPUT SCHEMA:
{{
  "relations": [
    {{
      "relation_type": "TYPE_NAME",
      "source_entity": "entity name",
      "target_entity": "entity name",
      "confidence": 0.0-1.0,
      "evidence": "quote from text supporting this relation"
    }}
  ]
}}

RULES:
- The text may be in ANY language. Extract relationships regardless of source language.
- Output all entity names in English, matching the names from the entity extraction step.
- Keep evidence quotes in the original language of the source text.
- Use entity NAMES (not IDs) for source_entity and target_entity
- Only extract explicit relationships stated in the text
- Do not infer relationships from co-occurrence alone
- If no relations found, return {{"relations": []}}

Document: {document_id}

ENTITIES:
{entities_json}

TEXT:
{text}

OUTPUT JSON:"""
