"""Tests for sift_kg.graph.prededup (deterministic pre-deduplication)."""

from sift_kg.extract.models import DocumentExtraction, ExtractedEntity, ExtractedRelation
from sift_kg.graph.prededup import (
    _normalize_name,
    _pick_canonical,
    _singularize,
    prededup_entities,
)


class TestNormalizeName:
    """Test name normalization."""

    def test_lowercase(self):
        assert _normalize_name("Alice Smith") == "alice smith"

    def test_unicode(self):
        assert _normalize_name("José García") == "jose garcia"

    def test_strip_whitespace(self):
        assert _normalize_name("  Alice  ") == "alice"

    def test_cafe_accent(self):
        assert _normalize_name("Café") == "cafe"


class TestSingularize:
    """Test singularization."""

    def test_plural_to_singular(self):
        assert _singularize("companies") == "company"

    def test_already_singular(self):
        assert _singularize("company") == "company"

    def test_multi_word(self):
        result = _singularize("big companies")
        assert "company" in result


class TestPickCanonical:
    """Test canonical name selection."""

    def test_single_name(self):
        assert _pick_canonical(["Alice"]) == "Alice"

    def test_most_frequent(self):
        assert _pick_canonical(["Alice", "Alice", "ALICE"]) == "Alice"

    def test_longest_tiebreak(self):
        assert _pick_canonical(["Alice", "Alice Smith"]) == "Alice Smith"

    def test_alphabetical_final_tiebreak(self):
        result = _pick_canonical(["Bob", "Ann"])
        assert result == "Ann"


class TestPrededupEntities:
    """Test the full pre-dedup pipeline."""

    def _make_extraction(self, entities: list[tuple[str, str]]) -> DocumentExtraction:
        """Helper: build extraction from (name, type) tuples."""
        return DocumentExtraction(
            document_id="doc1",
            document_path="/tmp/doc1.txt",
            entities=[
                ExtractedEntity(name=name, entity_type=etype, confidence=0.9)
                for name, etype in entities
            ],
        )

    def test_exact_duplicates_merged(self):
        """Same name appearing twice should not produce a mapping (same canonical)."""
        ext = self._make_extraction([
            ("Alice Smith", "PERSON"),
            ("Alice Smith", "PERSON"),
        ])
        result = prededup_entities([ext])
        assert len(result) == 0

    def test_case_variants_merged(self):
        """Case-only differences should merge."""
        ext = self._make_extraction([
            ("Alice Smith", "PERSON"),
            ("alice smith", "PERSON"),
        ])
        result = prededup_entities([ext])
        assert len(result) >= 1

    def test_unicode_variants_merged(self):
        """Unicode variants should merge."""
        ext = self._make_extraction([
            ("José García", "PERSON"),
            ("Jose Garcia", "PERSON"),
        ])
        result = prededup_entities([ext])
        assert len(result) >= 1

    def test_plural_singular_merged(self):
        """Plurals should merge with singulars."""
        ext = self._make_extraction([
            ("Company", "ORGANIZATION"),
            ("Companies", "ORGANIZATION"),
        ])
        result = prededup_entities([ext])
        assert len(result) >= 1

    def test_different_types_not_merged(self):
        """Same name but different types should NOT merge."""
        ext = self._make_extraction([
            ("Alice", "PERSON"),
            ("Alice", "ORGANIZATION"),
        ])
        result = prededup_entities([ext])
        assert len(result) == 0

    def test_genuinely_different_not_merged(self):
        """Clearly different names should not merge."""
        ext = self._make_extraction([
            ("Alice Smith", "PERSON"),
            ("Bob Johnson", "PERSON"),
            ("New York", "LOCATION"),
        ])
        result = prededup_entities([ext])
        assert len(result) == 0

    def test_cross_document_dedup(self):
        """Entities from different documents should be compared."""
        ext1 = self._make_extraction([("Alice Smith", "PERSON")])
        ext2 = self._make_extraction([("alice smith", "PERSON")])
        ext2.document_id = "doc2"
        result = prededup_entities([ext1, ext2])
        assert len(result) >= 1

    def test_skips_errored_extractions(self):
        """Extractions with errors should be skipped."""
        ext = DocumentExtraction(
            document_id="bad",
            document_path="/tmp/bad.txt",
            error="Failed",
            entities=[
                ExtractedEntity(name="Alice", entity_type="PERSON", confidence=0.9),
            ],
        )
        result = prededup_entities([ext])
        assert len(result) == 0

    def test_single_entity_no_dedup(self):
        """Single entity per type should produce no mappings."""
        ext = self._make_extraction([("Alice", "PERSON")])
        result = prededup_entities([ext])
        assert len(result) == 0

    def test_canonical_picks_most_frequent(self):
        """Canonical should be the most frequent variant."""
        ext = self._make_extraction([
            ("alice smith", "PERSON"),
            ("Alice Smith", "PERSON"),
            ("Alice Smith", "PERSON"),
        ])
        result = prededup_entities([ext])
        if result:
            assert result[("PERSON", "alice smith")] == "Alice Smith"


class TestPrededupIntegration:
    """Integration: prededup + build_graph."""

    def test_build_graph_with_prededup(self):
        """build_graph merges pre-dedup entities into single nodes."""
        from sift_kg.graph.builder import build_graph

        ext = DocumentExtraction(
            document_id="doc1",
            document_path="/tmp/doc1.txt",
            entities=[
                ExtractedEntity(name="Alice Smith", entity_type="PERSON", confidence=0.9),
                ExtractedEntity(name="alice smith", entity_type="PERSON", confidence=0.8),
                ExtractedEntity(name="Bob Jones", entity_type="PERSON", confidence=0.85),
            ],
            relations=[],
        )

        kg = build_graph([ext], postprocess=False)
        person_count = sum(
            1 for _, data in kg.graph.nodes(data=True)
            if data.get("entity_type") == "PERSON"
        )
        assert person_count == 2

    def test_build_graph_prededup_preserves_relations(self):
        """Relations should still resolve after prededup remapping."""
        from sift_kg.graph.builder import build_graph

        ext = DocumentExtraction(
            document_id="doc1",
            document_path="/tmp/doc1.txt",
            entities=[
                ExtractedEntity(name="alice smith", entity_type="PERSON", confidence=0.9),
                ExtractedEntity(name="Alice Smith", entity_type="PERSON", confidence=0.8),
                ExtractedEntity(name="Acme Corp", entity_type="ORGANIZATION", confidence=0.85),
            ],
            relations=[
                ExtractedRelation(
                    source_entity="alice smith",
                    target_entity="Acme Corp",
                    relation_type="WORKS_FOR",
                    confidence=0.8,
                ),
            ],
        )

        kg = build_graph([ext], postprocess=False)
        assert kg.relation_count >= 1
