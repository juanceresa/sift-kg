"""Pydantic models for domain configuration.

Defines the schema for domain YAML files that control what entity types
and relation types sift-kg extracts from documents.
"""


from pydantic import BaseModel, Field


class EntityTypeConfig(BaseModel):
    """Configuration for an entity type."""

    description: str = ""
    extraction_hints: list[str] = Field(default_factory=list)


class RelationTypeConfig(BaseModel):
    """Configuration for a relation type."""

    description: str = ""
    source_types: list[str] = Field(default_factory=list)
    target_types: list[str] = Field(default_factory=list)
    symmetric: bool = False
    extraction_hints: list[str] = Field(default_factory=list)
    review_required: bool = False  # Flag for analyst review


class DomainConfig(BaseModel):
    """Complete domain configuration loaded from YAML."""

    name: str
    version: str = "1.0.0"
    description: str = ""

    entity_types: dict[str, EntityTypeConfig] = Field(default_factory=dict)
    relation_types: dict[str, RelationTypeConfig] = Field(default_factory=dict)

    # Optional system context injected into LLM prompts
    system_context: str | None = None

    def get_entity_type_names(self) -> list[str]:
        """Get list of entity type names."""
        return list(self.entity_types.keys())

    def get_relation_type_names(self) -> list[str]:
        """Get list of relation type names."""
        return list(self.relation_types.keys())

    def get_extraction_hints(self, relation_type: str) -> list[str]:
        """Get extraction hints for a relation type."""
        config = self.relation_types.get(relation_type)
        return config.extraction_hints if config else []
