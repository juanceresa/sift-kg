"""Tests for sift_kg.resolve (models, io, engine)."""

import pytest
import yaml

from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.resolve.engine import apply_merges, apply_relation_rejections
from sift_kg.resolve.io import read_proposals, read_relation_review, write_proposals, write_relation_review
from sift_kg.resolve.models import (
    MergeFile,
    MergeMember,
    MergeProposal,
    RelationReviewEntry,
    RelationReviewFile,
)


class TestResolveModels:
    """Test merge and review Pydantic models."""

    def test_merge_member(self):
        """MergeMember stores id, name, confidence."""
        m = MergeMember(id="person:alice", name="Alice", confidence=0.9)
        assert m.id == "person:alice"

    def test_merge_proposal_status(self):
        """MergeProposal defaults to DRAFT."""
        p = MergeProposal(
            canonical_id="person:alice",
            canonical_name="Alice",
            entity_type="PERSON",
            members=[MergeMember(id="person:alice_smith", name="Alice Smith")],
        )
        assert p.status == "DRAFT"

    def test_merge_file_status_filters(self):
        """MergeFile filters proposals by status."""
        proposals = [
            MergeProposal(
                canonical_id="a",
                canonical_name="A",
                entity_type="PERSON",
                status="DRAFT",
                members=[MergeMember(id="b", name="B")],
            ),
            MergeProposal(
                canonical_id="c",
                canonical_name="C",
                entity_type="PERSON",
                status="CONFIRMED",
                members=[MergeMember(id="d", name="D")],
            ),
            MergeProposal(
                canonical_id="e",
                canonical_name="E",
                entity_type="PERSON",
                status="REJECTED",
                members=[MergeMember(id="f", name="F")],
            ),
        ]
        mf = MergeFile(proposals=proposals)
        assert len(mf.draft) == 1
        assert len(mf.confirmed) == 1
        assert len(mf.rejected) == 1

    def test_relation_review_status_filters(self):
        """RelationReviewFile filters entries by status."""
        entries = [
            RelationReviewEntry(
                source_id="a", source_name="A",
                target_id="b", target_name="B",
                relation_type="KNOWS",
                status="DRAFT",
            ),
            RelationReviewEntry(
                source_id="c", source_name="C",
                target_id="d", target_name="D",
                relation_type="KNOWS",
                status="REJECTED",
            ),
        ]
        rf = RelationReviewFile(relations=entries)
        assert len(rf.draft) == 1
        assert len(rf.rejected) == 1
        assert len(rf.confirmed) == 0


class TestResolveIO:
    """Test YAML read/write for merge proposals and relation reviews."""

    def test_write_read_proposals_roundtrip(self, tmp_dir):
        """Write and read merge proposals preserves data."""
        original = MergeFile(
            proposals=[
                MergeProposal(
                    canonical_id="person:alice",
                    canonical_name="Alice",
                    entity_type="PERSON",
                    status="CONFIRMED",
                    members=[MergeMember(id="person:alice_smith", name="Alice Smith", confidence=0.85)],
                    reason="Same person, different name format",
                ),
            ]
        )
        path = tmp_dir / "proposals.yaml"
        write_proposals(original, path)
        assert path.exists()

        loaded = read_proposals(path)
        assert len(loaded.proposals) == 1
        assert loaded.proposals[0].canonical_id == "person:alice"
        assert loaded.proposals[0].status == "CONFIRMED"
        assert loaded.proposals[0].members[0].name == "Alice Smith"

    def test_read_missing_proposals(self, tmp_dir):
        """Reading from nonexistent file returns empty MergeFile."""
        loaded = read_proposals(tmp_dir / "nonexistent.yaml")
        assert len(loaded.proposals) == 0

    def test_write_read_relation_review_roundtrip(self, tmp_dir):
        """Write and read relation reviews preserves data."""
        original = RelationReviewFile(
            review_threshold=0.6,
            relations=[
                RelationReviewEntry(
                    source_id="a", source_name="A",
                    target_id="b", target_name="B",
                    relation_type="WORKS_FOR",
                    confidence=0.4,
                    status="REJECTED",
                    flag_reason="Low confidence",
                ),
            ],
        )
        path = tmp_dir / "review.yaml"
        write_relation_review(original, path)
        assert path.exists()

        loaded = read_relation_review(path)
        assert loaded.review_threshold == 0.6
        assert len(loaded.relations) == 1
        assert loaded.relations[0].status == "REJECTED"

    def test_read_missing_relation_review(self, tmp_dir):
        """Reading from nonexistent file returns empty RelationReviewFile."""
        loaded = read_relation_review(tmp_dir / "nonexistent.yaml")
        assert len(loaded.relations) == 0


class TestApplyMerges:
    """Test entity merge application."""

    def _build_graph_with_duplicates(self):
        """Helper: graph with Alice and Alice Smith (same person)."""
        kg = KnowledgeGraph()
        kg.add_entity("person:alice", "PERSON", "Alice", confidence=0.9, source_documents=["doc1"])
        kg.add_entity("person:alice_smith", "PERSON", "Alice Smith", confidence=0.7, source_documents=["doc2"])
        kg.add_entity("org:acme", "ORGANIZATION", "Acme")

        kg.add_relation("r1", "person:alice", "org:acme", "WORKS_FOR")
        kg.add_relation("r2", "person:alice_smith", "org:acme", "WORKS_FOR")
        return kg

    def test_merge_removes_member_node(self):
        """Merging removes the member node."""
        kg = self._build_graph_with_duplicates()
        merge_file = MergeFile(proposals=[
            MergeProposal(
                canonical_id="person:alice",
                canonical_name="Alice",
                entity_type="PERSON",
                status="CONFIRMED",
                members=[MergeMember(id="person:alice_smith", name="Alice Smith")],
            ),
        ])
        stats = apply_merges(kg, merge_file)
        assert stats["nodes_removed"] == 1
        assert kg.get_entity("person:alice_smith") is None
        assert kg.get_entity("person:alice") is not None

    def test_merge_rewrites_edges(self):
        """Edges from member node are rewritten to canonical."""
        kg = self._build_graph_with_duplicates()
        merge_file = MergeFile(proposals=[
            MergeProposal(
                canonical_id="person:alice",
                canonical_name="Alice",
                entity_type="PERSON",
                status="CONFIRMED",
                members=[MergeMember(id="person:alice_smith", name="Alice Smith")],
            ),
        ])
        apply_merges(kg, merge_file)
        # Alice should have relations to Acme (her original + rewritten from Alice Smith)
        relations = kg.get_relations("person:alice", direction="out")
        works_for = [r for r in relations if r.get("relation_type") == "WORKS_FOR"]
        assert len(works_for) >= 1

    def test_merge_combines_source_documents(self):
        """Merge combines source_documents from both entities."""
        kg = self._build_graph_with_duplicates()
        merge_file = MergeFile(proposals=[
            MergeProposal(
                canonical_id="person:alice",
                canonical_name="Alice",
                entity_type="PERSON",
                status="CONFIRMED",
                members=[MergeMember(id="person:alice_smith", name="Alice Smith")],
            ),
        ])
        apply_merges(kg, merge_file)
        entity = kg.get_entity("person:alice")
        assert "doc1" in entity["source_documents"]
        assert "doc2" in entity["source_documents"]

    def test_merge_keeps_higher_confidence(self):
        """Merge keeps the higher confidence value."""
        kg = self._build_graph_with_duplicates()
        merge_file = MergeFile(proposals=[
            MergeProposal(
                canonical_id="person:alice",
                canonical_name="Alice",
                entity_type="PERSON",
                status="CONFIRMED",
                members=[MergeMember(id="person:alice_smith", name="Alice Smith")],
            ),
        ])
        apply_merges(kg, merge_file)
        entity = kg.get_entity("person:alice")
        assert entity["confidence"] == 0.9

    def test_merge_removes_self_loops(self):
        """Self-loops created by merging are removed."""
        kg = KnowledgeGraph()
        kg.add_entity("a", "PERSON", "A")
        kg.add_entity("b", "PERSON", "B")
        # A knows B — after merge, this becomes A knows A (self-loop)
        kg.add_relation("r1", "a", "b", "KNOWS")

        merge_file = MergeFile(proposals=[
            MergeProposal(
                canonical_id="a",
                canonical_name="A",
                entity_type="PERSON",
                status="CONFIRMED",
                members=[MergeMember(id="b", name="B")],
            ),
        ])
        stats = apply_merges(kg, merge_file)
        assert stats["self_loops_removed"] == 1
        assert kg.relation_count == 0

    def test_draft_proposals_not_applied(self):
        """Only CONFIRMED proposals are applied, not DRAFT."""
        kg = self._build_graph_with_duplicates()
        merge_file = MergeFile(proposals=[
            MergeProposal(
                canonical_id="person:alice",
                canonical_name="Alice",
                entity_type="PERSON",
                status="DRAFT",
                members=[MergeMember(id="person:alice_smith", name="Alice Smith")],
            ),
        ])
        stats = apply_merges(kg, merge_file)
        assert stats["merges_applied"] == 0
        assert kg.get_entity("person:alice_smith") is not None  # Still exists

    def test_empty_merge_file(self):
        """Empty merge file does nothing."""
        kg = KnowledgeGraph()
        kg.add_entity("a", "PERSON", "A")
        stats = apply_merges(kg, MergeFile())
        assert stats["merges_applied"] == 0


class TestApplyRelationRejections:
    """Test relation rejection application."""

    def test_reject_removes_relation(self):
        """REJECTED relations are removed from graph."""
        kg = KnowledgeGraph()
        kg.add_entity("a", "PERSON", "A")
        kg.add_entity("b", "ORG", "B")
        kg.add_relation("r1", "a", "b", "WORKS_FOR", confidence=0.3)

        review = RelationReviewFile(relations=[
            RelationReviewEntry(
                source_id="a", source_name="A",
                target_id="b", target_name="B",
                relation_type="WORKS_FOR",
                status="REJECTED",
            ),
        ])
        removed = apply_relation_rejections(kg, review)
        assert removed == 1
        assert kg.relation_count == 0

    def test_confirmed_relations_kept(self):
        """CONFIRMED relations are not removed."""
        kg = KnowledgeGraph()
        kg.add_entity("a", "PERSON", "A")
        kg.add_entity("b", "ORG", "B")
        kg.add_relation("r1", "a", "b", "WORKS_FOR")

        review = RelationReviewFile(relations=[
            RelationReviewEntry(
                source_id="a", source_name="A",
                target_id="b", target_name="B",
                relation_type="WORKS_FOR",
                status="CONFIRMED",
            ),
        ])
        removed = apply_relation_rejections(kg, review)
        assert removed == 0
        assert kg.relation_count == 1

    def test_empty_review_no_changes(self):
        """Empty review file removes nothing."""
        kg = KnowledgeGraph()
        kg.add_entity("a", "PERSON", "A")
        kg.add_entity("b", "ORG", "B")
        kg.add_relation("r1", "a", "b", "WORKS_FOR")

        removed = apply_relation_rejections(kg, RelationReviewFile())
        assert removed == 0
        assert kg.relation_count == 1
