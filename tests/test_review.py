"""Tests for OSINT domain and interactive reviewer."""

from unittest.mock import patch

from sift_kg.domains.loader import DomainLoader
from sift_kg.resolve.models import (
    MergeFile,
    MergeMember,
    MergeProposal,
    RelationReviewEntry,
    RelationReviewFile,
)
from sift_kg.resolve.reviewer import review_merges, review_relations


class TestOsintDomain:
    """Test the bundled OSINT domain."""

    def test_osint_in_bundled_list(self):
        """OSINT domain appears in bundled domain list."""
        loader = DomainLoader()
        available = loader.list_bundled()
        assert "osint" in available

    def test_both_domains_available(self):
        """Both default and osint are available."""
        loader = DomainLoader()
        available = loader.list_bundled()
        assert "default" in available
        assert "osint" in available

    def test_osint_loads_successfully(self):
        """OSINT domain loads without errors."""
        loader = DomainLoader()
        domain = loader.load_bundled("osint")
        assert domain.name == "OSINT Investigation"

    def test_osint_entity_types(self):
        """OSINT domain has expected entity types."""
        loader = DomainLoader()
        domain = loader.load_bundled("osint")
        expected = {"PERSON", "ORGANIZATION", "SHELL_COMPANY", "FINANCIAL_ACCOUNT", "LOCATION", "DOCUMENT", "EVENT"}
        assert set(domain.entity_types.keys()) == expected

    def test_osint_relation_types(self):
        """OSINT domain has expected relation types."""
        loader = DomainLoader()
        domain = loader.load_bundled("osint")
        expected = {
            "BENEFICIAL_OWNER_OF", "DIRECTOR_OF", "SHAREHOLDER_OF",
            "TRANSACTED_WITH", "SUBSIDIARY_OF", "REGISTERED_IN",
            "SIGNATORY_OF", "ASSOCIATED_WITH", "LOCATED_IN", "MENTIONED_IN",
        }
        assert set(domain.relation_types.keys()) == expected

    def test_osint_review_required_relations(self):
        """BENEFICIAL_OWNER_OF and TRANSACTED_WITH require review."""
        loader = DomainLoader()
        domain = loader.load_bundled("osint")
        assert domain.relation_types["BENEFICIAL_OWNER_OF"].review_required is True
        assert domain.relation_types["TRANSACTED_WITH"].review_required is True
        assert domain.relation_types["DIRECTOR_OF"].review_required is False

    def test_osint_has_system_context(self):
        """OSINT domain includes system context for LLM."""
        loader = DomainLoader()
        domain = loader.load_bundled("osint")
        assert domain.system_context is not None
        assert "intelligence" in domain.system_context.lower()


class TestReviewMerges:
    """Test interactive merge review (with mocked input)."""

    def _make_merge_file(self, n=3):
        """Create a MergeFile with n DRAFT proposals."""
        proposals = [
            MergeProposal(
                canonical_id=f"person:p{i}",
                canonical_name=f"Person {i}",
                entity_type="PERSON",
                status="DRAFT",
                members=[MergeMember(id=f"person:p{i}_alt", name=f"P{i} Alt")],
                reason=f"Same person {i}",
            )
            for i in range(n)
        ]
        return MergeFile(proposals=proposals)

    @patch("builtins.input", side_effect=["a", "a", "a"])
    def test_approve_all(self, mock_input):
        """Approving all sets status to CONFIRMED."""
        mf = self._make_merge_file(3)
        stats = review_merges(mf)
        assert stats["approved"] == 3
        assert stats["rejected"] == 0
        assert stats["skipped"] == 0
        assert all(p.status == "CONFIRMED" for p in mf.proposals)

    @patch("builtins.input", side_effect=["r", "r"])
    def test_reject_all(self, mock_input):
        """Rejecting all sets status to REJECTED."""
        mf = self._make_merge_file(2)
        stats = review_merges(mf)
        assert stats["rejected"] == 2
        assert all(p.status == "REJECTED" for p in mf.proposals)

    @patch("builtins.input", side_effect=["a", "r", "s"])
    def test_mixed_decisions(self, mock_input):
        """Mixed approve/reject/skip updates statuses correctly."""
        mf = self._make_merge_file(3)
        stats = review_merges(mf)
        assert stats["approved"] == 1
        assert stats["rejected"] == 1
        assert stats["skipped"] == 1
        assert mf.proposals[0].status == "CONFIRMED"
        assert mf.proposals[1].status == "REJECTED"
        assert mf.proposals[2].status == "DRAFT"  # skipped stays DRAFT

    @patch("builtins.input", side_effect=["a", "q"])
    def test_quit_skips_remaining(self, mock_input):
        """Quitting skips all remaining proposals."""
        mf = self._make_merge_file(3)
        stats = review_merges(mf)
        assert stats["approved"] == 1
        assert stats["skipped"] == 2  # quit skips remaining
        assert mf.proposals[0].status == "CONFIRMED"
        assert mf.proposals[1].status == "DRAFT"  # untouched
        assert mf.proposals[2].status == "DRAFT"

    def test_no_drafts_skips(self):
        """No DRAFT proposals returns zero counts."""
        mf = MergeFile(proposals=[
            MergeProposal(
                canonical_id="a", canonical_name="A", entity_type="PERSON",
                status="CONFIRMED",
                members=[MergeMember(id="b", name="B")],
            ),
        ])
        stats = review_merges(mf)
        assert stats["approved"] == 0
        assert stats["rejected"] == 0
        assert stats["skipped"] == 0


class TestReviewRelations:
    """Test interactive relation review (with mocked input)."""

    def _make_review_file(self, n=2):
        """Create a RelationReviewFile with n DRAFT entries."""
        entries = [
            RelationReviewEntry(
                source_id=f"a{i}", source_name=f"A{i}",
                target_id=f"b{i}", target_name=f"B{i}",
                relation_type="WORKS_FOR",
                confidence=0.4,
                status="DRAFT",
                flag_reason="Low confidence",
            )
            for i in range(n)
        ]
        return RelationReviewFile(relations=entries)

    @patch("builtins.input", side_effect=["a", "r"])
    def test_approve_and_reject(self, mock_input):
        """Mixed approve/reject on relations."""
        rf = self._make_review_file(2)
        stats = review_relations(rf)
        assert stats["approved"] == 1
        assert stats["rejected"] == 1
        assert rf.relations[0].status == "CONFIRMED"
        assert rf.relations[1].status == "REJECTED"

    @patch("builtins.input", side_effect=["q"])
    def test_quit_relations(self, mock_input):
        """Quitting skips all remaining relations."""
        rf = self._make_review_file(3)
        stats = review_relations(rf)
        assert stats["skipped"] == 3

    def test_no_draft_relations(self):
        """No DRAFT relations returns zero counts."""
        rf = RelationReviewFile(relations=[
            RelationReviewEntry(
                source_id="a", source_name="A",
                target_id="b", target_name="B",
                relation_type="KNOWS", status="CONFIRMED",
            ),
        ])
        stats = review_relations(rf)
        assert stats == {"approved": 0, "rejected": 0, "skipped": 0}
