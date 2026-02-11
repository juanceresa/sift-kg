"""Pydantic models for merge proposals and relation reviews.

Two review systems:
  1. Entity merges — LLM proposes, user confirms/rejects
  2. Relation reviews — flagged relations user confirms/rejects
"""

from typing import Literal

from pydantic import BaseModel, Field

StatusType = Literal["DRAFT", "CONFIRMED", "REJECTED"]


# ============================================================================
# Entity Merge Models
# ============================================================================


class MergeMember(BaseModel):
    """An entity that should be merged into a canonical entity."""

    id: str
    name: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MergeProposal(BaseModel):
    """A proposed merge of multiple entities into one canonical entity.

    The `members` list contains the non-canonical entities that will be merged
    INTO the canonical entity. The canonical entity itself is identified by
    `canonical_id` and is not included in `members`.
    """

    canonical_id: str
    canonical_name: str
    entity_type: str
    status: StatusType = "DRAFT"
    members: list[MergeMember] = Field(min_length=1)
    reason: str = ""  # LLM's explanation for why these should merge


class MergeFile(BaseModel):
    """Top-level model for merge_proposals.yaml."""

    proposals: list[MergeProposal] = Field(default_factory=list)

    @property
    def confirmed(self) -> list[MergeProposal]:
        return [p for p in self.proposals if p.status == "CONFIRMED"]

    @property
    def draft(self) -> list[MergeProposal]:
        return [p for p in self.proposals if p.status == "DRAFT"]

    @property
    def rejected(self) -> list[MergeProposal]:
        return [p for p in self.proposals if p.status == "REJECTED"]


# ============================================================================
# Relation Review Models
# ============================================================================


class RelationReviewEntry(BaseModel):
    """A flagged relation requiring user review."""

    source_id: str
    source_name: str
    target_id: str
    target_name: str
    relation_type: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = ""
    source_document: str = ""
    status: StatusType = "DRAFT"
    flag_reason: str = ""  # Why this was flagged


class RelationReviewFile(BaseModel):
    """Top-level model for relation_review.yaml."""

    review_threshold: float = 0.7  # Relations below this confidence are flagged
    relations: list[RelationReviewEntry] = Field(default_factory=list)

    @property
    def confirmed(self) -> list[RelationReviewEntry]:
        return [r for r in self.relations if r.status == "CONFIRMED"]

    @property
    def draft(self) -> list[RelationReviewEntry]:
        return [r for r in self.relations if r.status == "DRAFT"]

    @property
    def rejected(self) -> list[RelationReviewEntry]:
        return [r for r in self.relations if r.status == "REJECTED"]
