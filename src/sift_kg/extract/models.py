"""Pydantic models for extraction results.

Generic models — entity_type is a string driven by domain config,
not concrete Person/Property/Organization subclasses.
"""

from typing import Any

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """An entity extracted from a document chunk."""

    name: str
    entity_type: str  # Driven by domain config (PERSON, ORGANIZATION, etc.)
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    context: str = ""  # Quote from source text


class ExtractedRelation(BaseModel):
    """A relation between two entities extracted from a document chunk."""

    relation_type: str
    source_entity: str  # Entity name
    target_entity: str  # Entity name
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = ""  # Quote from source text


class ExtractionResult(BaseModel):
    """Complete extraction result for one document chunk."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    source_document: str = ""
    chunk_index: int = 0


class DocumentExtraction(BaseModel):
    """Complete extraction for an entire document (all chunks merged)."""

    document_id: str
    document_path: str
    chunks_processed: int = 0
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    cost_usd: float = 0.0
    model_used: str = ""
    error: str | None = None
    # Incremental extraction metadata (defaults for backward compat with existing JSONs)
    domain_name: str = ""
    chunk_size: int = 0
    extracted_at: str = ""
