"""LLM-driven schema discovery for schema-free extraction.

Samples documents, asks an LLM to design entity and relation types,
then saves the result as a reusable domain YAML. This eliminates
cross-chunk type drift while keeping the "no predefined schema" UX.
"""

import logging
from pathlib import Path

import yaml

from sift_kg.domains.loader import DomainLoader
from sift_kg.domains.models import DomainConfig, EntityTypeConfig, RelationTypeConfig

logger = logging.getLogger(__name__)

# Truncate each sample to this many characters
_SAMPLE_MAX_CHARS = 3000


def build_discovery_prompt(
    samples: list[str],
    system_context: str = "",
) -> str:
    """Build a prompt asking the LLM to design a schema for this corpus.

    Args:
        samples: Text samples (first chunk from up to 5 docs, pre-truncated)
        system_context: Optional domain context from domain config
    """
    context_section = ""
    if system_context:
        context_section = f"\n{system_context}\n"

    sample_sections = []
    for i, sample in enumerate(samples, 1):
        sample_sections.append(f"--- SAMPLE {i} ---\n{sample[:_SAMPLE_MAX_CHARS]}")
    samples_text = "\n\n".join(sample_sections)

    return f"""{context_section}You are a knowledge graph architect. Analyze the document samples below and design a schema (entity types + relation types) that best captures the information in this corpus.

DOCUMENT SAMPLES:
{samples_text}

TASK:
Design a schema with:
- 5-15 entity types that cover the key concepts in these documents
- 8-20 relation types that capture the important relationships

REQUIREMENTS:
- All type names must be UPPERCASE_SNAKE_CASE (e.g. PERSON, GOVERNMENT_AGENCY, FUNDED_BY)
- Each entity type needs a brief description
- Each relation type needs a description and source_types / target_types lists
- Be specific: prefer UNIVERSITY over ORGANIZATION, COURT_CASE over EVENT when the data supports it
- Relation types should use active voice: EMPLOYED_BY not EMPLOYMENT, FOUNDED not FOUNDING_OF

Return ONLY valid JSON matching this schema:
{{
  "entity_types": {{
    "TYPE_NAME": {{
      "description": "what this type represents"
    }}
  }},
  "relation_types": {{
    "RELATION_NAME": {{
      "description": "what this relation means",
      "source_types": ["ENTITY_TYPE"],
      "target_types": ["ENTITY_TYPE"]
    }}
  }}
}}

OUTPUT JSON:"""


async def discover_domain(
    samples: list[str],
    llm: "LLMClient",  # noqa: F821
    system_context: str = "",
) -> DomainConfig:
    """Run schema discovery via LLM and return a concrete DomainConfig.

    Args:
        samples: Text samples from documents
        llm: LLM client instance
        system_context: Optional domain context

    Returns:
        DomainConfig with discovered entity/relation types (schema_free=False)

    Raises:
        RuntimeError, ValueError: On LLM or parse failure (callers handle fallback)
    """
    prompt = build_discovery_prompt(samples, system_context)
    data = await llm.acall_json(prompt)

    entity_types: dict[str, EntityTypeConfig] = {}
    for name, cfg in data.get("entity_types", {}).items():
        if isinstance(cfg, str):
            entity_types[name.upper()] = EntityTypeConfig(description=cfg)
        elif isinstance(cfg, dict):
            entity_types[name.upper()] = EntityTypeConfig(
                description=cfg.get("description", ""),
                extraction_hints=cfg.get("extraction_hints", []),
            )

    relation_types: dict[str, RelationTypeConfig] = {}
    for name, cfg in data.get("relation_types", {}).items():
        if isinstance(cfg, str):
            relation_types[name.upper()] = RelationTypeConfig(description=cfg)
        elif isinstance(cfg, dict):
            relation_types[name.upper()] = RelationTypeConfig(
                description=cfg.get("description", ""),
                source_types=cfg.get("source_types", []),
                target_types=cfg.get("target_types", []),
                symmetric=cfg.get("symmetric", False),
            )

    if not entity_types:
        raise ValueError("Discovery returned no entity types")

    return DomainConfig(
        name="Auto-Discovered",
        version="1.0.0",
        description="Schema discovered by LLM from document samples",
        entity_types=entity_types,
        relation_types=relation_types,
        schema_free=False,
    )


def save_discovered_domain(domain: DomainConfig, path: Path) -> None:
    """Serialize a DomainConfig to YAML matching DomainLoader format."""
    data: dict = {
        "name": domain.name,
        "version": domain.version,
        "description": domain.description,
        "schema_free": domain.schema_free,
        "entity_types": {},
        "relation_types": {},
    }
    if domain.system_context:
        data["system_context"] = domain.system_context

    for name, cfg in domain.entity_types.items():
        entry: dict = {"description": cfg.description}
        if cfg.extraction_hints:
            entry["extraction_hints"] = cfg.extraction_hints
        if cfg.canonical_names:
            entry["canonical_names"] = cfg.canonical_names
        if cfg.canonical_fallback_type:
            entry["canonical_fallback_type"] = cfg.canonical_fallback_type
        data["entity_types"][name] = entry

    for name, cfg in domain.relation_types.items():
        entry = {"description": cfg.description}
        if cfg.source_types:
            entry["source_types"] = cfg.source_types
        if cfg.target_types:
            entry["target_types"] = cfg.target_types
        if cfg.symmetric:
            entry["symmetric"] = cfg.symmetric
        if cfg.extraction_hints:
            entry["extraction_hints"] = cfg.extraction_hints
        if cfg.review_required:
            entry["review_required"] = cfg.review_required
        data["relation_types"][name] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    logger.info(f"Saved discovered domain to {path}")


def load_discovered_domain(path: Path) -> DomainConfig | None:
    """Load a previously discovered domain, or None if missing/corrupt."""
    if not path.exists():
        return None
    try:
        loader = DomainLoader()
        return loader.load_from_path(path)
    except Exception as e:
        logger.warning(f"Failed to load discovered domain from {path}: {e}")
        return None
