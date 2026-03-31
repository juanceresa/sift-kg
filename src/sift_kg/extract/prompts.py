"""LLM提取提示词 — 域驱动，零样本。

组合提示在一个LLM调用中同时提取实体和关系，每个分块只需要一次调用。
"""

from sift_kg.domains.models import DomainConfig


def build_combined_prompt(
    text: str,
    document_id: str,
    domain: DomainConfig,
    doc_context: str = "",
) -> str:
    """构建同时提取实体和关系的单个提示词。

    将每个分块的LLM调用从2次减少到1次。提示词 instructs 模型
    先识别实体，然后在它们之间寻找关系。

    Args:
        doc_context: 可选的文档级摘要，预加到每个分块提示中，
            让LLM获得整个文档的上下文。
    """
    context_section = ""
    if domain.system_context:
        # 添加域提供的系统上下文
        context_section = f"\n{domain.system_context}\n"

    doc_context_section = ""
    if doc_context:
        # 添加文档级上下文
        doc_context_section = (
            "\nDOCUMENT CONTEXT (applies to entire document, not just this excerpt):\n"
            f"{doc_context}\n"
        )

    if domain.schema_free:
        # 无模式域，使用无模式提示
        return _build_schema_free_prompt(
            text,
            document_id,
            domain,
            context_section,
            doc_context_section,
        )

    # 构建实体类型列表
    type_lines = []
    for name, cfg in domain.entity_types.items():
        desc = cfg.description or name
        hints = ""
        if cfg.extraction_hints:
            # 添加提取提示
            hints = " (" + "; ".join(cfg.extraction_hints) + ")"
        if cfg.canonical_names:
            # 如果有规范名称列表，添加到提示中强制要求只使用这些名称
            names_list = ", ".join(cfg.canonical_names)
            hints += f" ALLOWED VALUES (use ONLY these exact names): [{names_list}]"
        type_lines.append(f"- {name}: {desc}{hints}")

    entity_types_section = "\n".join(type_lines)
    rel_types = ", ".join(domain.relation_types.keys())
    fallback = domain.fallback_relation

    # 为有源/目标约束的关系类型构建方向提示
    direction_lines = []
    for rel_name, rel_cfg in domain.relation_types.items():
        if rel_cfg.source_types and rel_cfg.target_types:
            src = "/".join(rel_cfg.source_types)
            tgt = "/".join(rel_cfg.target_types)
            direction_lines.append(f"- {rel_name}: {src} → {tgt}")

    direction_section = ""
    if direction_lines:
        # 添加方向说明，强制要求源实体类型必须在左侧，目标在右侧
        direction_section = (
            "\n\nRELATION DIRECTIONS (source_entity → target_entity):\n"
            + "\n".join(direction_lines)
            + "\nIMPORTANT: source_entity must be the type on the LEFT, target_entity the type on the RIGHT."
        )

    return f"""{context_section}Extract entities and relationships from the following document text. Return valid JSON only.

STEP 1 — ENTITIES
Identify all entities in the text.

ENTITY TYPES (use ONLY these — do not invent new types):
{entity_types_section}
IMPORTANT: Every entity must use one of the types listed above. Do not create new entity types.

STEP 2 — RELATIONS
Identify relationships between the entities you extracted.

RELATION TYPES (use ONLY these — do not invent new types): {rel_types}
{f"If a relationship doesn't fit any listed type, use {fallback}." if fallback else "Only extract relationships that clearly match one of the listed types."}{direction_section}

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
- Extract ALL entities that match the defined types, not just the most prominent
- Extract only explicit information from the text
- The text may be in ANY language. Extract entities regardless of source language.
- Output all entity names and attribute values in English. Use the most internationally recognized form of each name — do not anglicize personal names (Juan stays Juan, not John; 习近平 → Xi Jinping, not "Xi Near-Peace").
- Keep context and evidence quotes in the original language of the source text.
- confidence: 0.0-1.0 based on how clearly the text supports the extraction and how well it fits the assigned type
- attributes: include any relevant details (dates, roles, descriptions, etc.)
- Use entity NAMES (not IDs) for source_entity and target_entity
- Only extract relationships that are explicitly stated in the text and match a defined relation type
- Do not infer relationships from co-occurrence alone
- If no relations found, return an empty relations list

Document: {document_id}
{doc_context_section}
TEXT:
{text}

OUTPUT JSON:"""


def _build_schema_free_prompt(
    text: str,
    document_id: str,
    domain: DomainConfig,
    context_section: str,
    doc_context_section: str,
) -> str:
    """构建无模式提取的组合提示词。

    LLM从数据中有机地发现实体和关系类型，
    而不是被约束到预定义的模式。
    """
    # 如果用户在自定义无模式域（混合）中提供了实体类型提示，
    # 将它们作为指导而不是约束包含进去
    entity_guidance = ""
    if domain.entity_types:
        hint_lines = []
        for name, cfg in domain.entity_types.items():
            desc = cfg.description or name
            hints = ""
            if cfg.extraction_hints:
                # 添加提取提示
                hints = " (" + "; ".join(cfg.extraction_hints) + ")"
            hint_lines.append(f"- {name}: {desc}{hints}")
        entity_guidance = (
            "\nSuggested entity types (use these when they fit, but also discover new types as needed):\n"
            + "\n".join(hint_lines)
            + "\n"
        )

    return f"""{context_section}Extract entities and relationships from the following document text. Return valid JSON only.

STEP 1 — ENTITIES
Identify all entities in the text. Determine the most descriptive entity type for each.
{entity_guidance}
ENTITY TYPE GUIDELINES:
- Use UPPERCASE_SNAKE_CASE for all entity types (e.g. PERSON, COMPANY, LEGAL_CASE, FINANCIAL_INSTRUMENT, GOVERNMENT_AGENCY)
- Be specific: prefer UNIVERSITY over ORGANIZATION, COURT_CASE over EVENT, BANK over COMPANY — when the data supports it
- Be consistent: use the same type name for similar entities across the text
- Common types include: PERSON, ORGANIZATION, LOCATION, EVENT, DOCUMENT, DATE, CONCEPT — but use whatever fits the data best

STEP 2 — RELATIONS
Identify relationships between the entities you extracted.

RELATION TYPE GUIDELINES:
- Use UPPERCASE_SNAKE_CASE for all relation types (e.g. EMPLOYED_BY, FUNDED, TESTIFIED_AGAINST, LOCATED_IN)
- Be specific and descriptive: prefer FUNDED over ASSOCIATED_WITH, TESTIFIED_AGAINST over RELATED_TO
- Use active voice: EMPLOYED_BY not EMPLOYMENT, FOUNDED not FOUNDING_OF
- Be consistent: use the same relation type for similar relationships

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
- Extract ALL entities present in the text, not just the most prominent
- Extract only explicit information from the text
- The text may be in ANY language. Extract entities regardless of source language.
- Output all entity names and attribute values in English. Use the most internationally recognized form of each name — do not anglicize personal names (Juan stays Juan, not John; 习近平 → Xi Jinping, not "Xi Near-Peace").
- Keep context and evidence quotes in the original language of the source text.
- confidence: 0.0-1.0 based on how clearly the text supports the extraction
- attributes: include any relevant details (dates, roles, descriptions, etc.)
- Use entity NAMES (not IDs) for source_entity and target_entity
- Only extract relationships that are explicitly stated in the text
- Do not infer relationships from co-occurrence alone
- If no relations found, return an empty relations list

Document: {document_id}
{doc_context_section}
TEXT:
{text}

OUTPUT JSON:"""
