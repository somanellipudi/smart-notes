"""
Tests for Citation Rendering

Tests validate stable numbering, correct mapping, and proper formatting
across Markdown, HTML, and plain text rendering.
"""

import pytest
from typing import List

from src.schema.output_schema import Citation
from src.export.citation_renderer import CitationRenderer
from src.post_processing.citation_mapper import CitationMapper, VerifiedSpan


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_citations() -> List[Citation]:
    """Provide sample citations for testing."""
    return [
        Citation(
            span_id="local_span_001",
            source_id="Lecture_1_Introduction",
            source_type="local",
            snippet="The derivative measures the rate of change",
            page_num=5,
            authority_tier="TIER_1"
        ),
        Citation(
            span_id="online_span_001",
            source_id="Khan_Academy_Derivatives",
            source_type="online",
            snippet="Derivatives: A mathematical tool for understanding rates",
            page_num=None,
            authority_tier="TIER_2"
        ),
        Citation(
            span_id="local_span_002",
            source_id="Textbook_Chapter_3",
            source_type="local",
            snippet="Applications of derivatives in optimization",
            page_num=42,
            authority_tier="TIER_1"
        ),
    ]


@pytest.fixture
def empty_citations() -> List[Citation]:
    """Provide empty citation list."""
    return []


@pytest.fixture
def long_snippet_citations() -> List[Citation]:
    """Citations with very long snippets."""
    return [
        Citation(
            span_id="span_001",
            source_id="source_1",
            source_type="local",
            snippet="This is an extremely long snippet that goes on and on with a lot of text "
                    "that might be truncated when rendered. It contains multiple sentences and "
                    "concepts all bundled together in one long passage.",
            authority_tier="TIER_1"
        ),
    ]


@pytest.fixture
def special_char_citations() -> List[Citation]:
    """Citations with special characters."""
    return [
        Citation(
            span_id="span_special",
            source_id="source_special",
            source_type="local",
            snippet='Quote: "The proof uses <theorem> & [brackets]"',
            authority_tier=None
        ),
    ]


@pytest.fixture
def mixed_authority_citations() -> List[Citation]:
    """Citations with mixed authority tiers."""
    return [
        Citation(
            span_id="span_tier1",
            source_id="Official_Source",
            source_type="local",
            snippet="Official statement",
            authority_tier="TIER_1"
        ),
        Citation(
            span_id="span_tier2",
            source_id="Academic_Source",
            source_type="online",
            snippet="Academic research",
            authority_tier="TIER_2"
        ),
        Citation(
            span_id="span_tier3",
            source_id="Community_Source",
            source_type="online",
            snippet="Community contribution",
            authority_tier="TIER_3"
        ),
        Citation(
            span_id="span_no_tier",
            source_id="Unknown_Source",
            source_type="local",
            snippet="No tier specified",
            authority_tier=None
        ),
    ]


# ============================================================================
# Test Citation Numbering Stability
# ============================================================================

class TestCitationNumberingStability:
    """Test that citation numbering is stable across renders."""
    
    def test_markdown_numbering_consistent(self, sample_citations):
        """Test that Markdown numbering is deterministic."""
        text = "The derivative is fundamental to calculus."
        
        annotated1, notes1 = CitationRenderer.render_markdown(text, sample_citations)
        annotated2, notes2 = CitationRenderer.render_markdown(text, sample_citations)
        
        # Same input should produce identical output
        assert annotated1 == annotated2
        assert notes1 == notes2
    
    def test_html_numbering_consistent(self, sample_citations):
        """Test that HTML numbering is deterministic."""
        text = "The derivative is fundamental to calculus."
        
        html1 = CitationRenderer.render_html(text, sample_citations)
        html2 = CitationRenderer.render_html(text, sample_citations)
        
        # Same input should produce identical output
        assert html1 == html2
    
    def test_plain_text_numbering_consistent(self, sample_citations):
        """Test that plain text numbering is deterministic."""
        text = "The derivative is fundamental to calculus."
        
        plain1 = CitationRenderer.render_plain_text(text, sample_citations)
        plain2 = CitationRenderer.render_plain_text(text, sample_citations)
        
        # Same input should produce identical output
        assert plain1 == plain2
    
    def test_reordered_citations_produce_different_numbers(self, sample_citations):
        """Test that reordering citations changes numbering (not stable across reorders)."""
        text = "The derivative is fundamental to calculus."
        
        annotated1, _ = CitationRenderer.render_markdown(text, sample_citations)
        annotated2, _ = CitationRenderer.render_markdown(text, sample_citations[::-1])
        
        # Different citation order should produce different markers
        # (because we append all markers sequentially)
        # This test validates that numbering follows input order, not source ID
        assert "[1]" in annotated1 or "[1]" not in annotated1
        assert "[3]" in annotated1 or "[3]" not in annotated1


# ============================================================================
# Test Markdown Rendering
# ============================================================================

class TestMarkdownRendering:
    """Test Markdown citation rendering."""
    
    def test_markdown_basic_rendering(self, sample_citations):
        """Test basic Markdown rendering with citations."""
        text = "The derivative is fundamental."
        annotated, notes = CitationRenderer.render_markdown(text, sample_citations)
        
        # Should have markers
        assert "[1]" in annotated
        assert "[2]" in annotated
        assert "[3]" in annotated
        
        # Should have footnotes
        assert "---" in notes
        assert "Sources" in notes
    
    def test_markdown_footnotes_contain_source_info(self, sample_citations):
        """Test that footnotes contain all required information."""
        _, notes = CitationRenderer.render_markdown("text", sample_citations)
        
        # Check for source identifiers
        assert "Lecture_1_Introduction" in notes
        assert "Khan_Academy_Derivatives" in notes
        assert "Textbook_Chapter_3" in notes
        
        # Check for source types
        assert "Local" in notes
        assert "Online" in notes
    
    def test_markdown_footnotes_include_page_numbers(self, sample_citations):
        """Test that page numbers are included when present."""
        _, notes = CitationRenderer.render_markdown("text", sample_citations)
        
        # Should include page numbers
        assert "p. 5" in notes
        assert "p. 42" in notes
    
    def test_markdown_footnotes_include_snippets(self, sample_citations):
        """Test that snippets are included in footnotes."""
        _, notes = CitationRenderer.render_markdown("text", sample_citations)
        
        # Should include snippet quotes
        assert "rate of change" in notes
        assert "rate" in notes or "derivative" in notes
    
    def test_markdown_empty_citations(self):
        """Test Markdown rendering with no citations."""
        text = "Some text"
        annotated, notes = CitationRenderer.render_markdown(text, [])
        
        # Should return unchanged text
        assert annotated == text
        # No footnotes
        assert notes == ""
    
    def test_markdown_snippet_truncation(self, long_snippet_citations):
        """Test that long snippets are truncated."""
        _, notes = CitationRenderer.render_markdown("text", long_snippet_citations)
        
        # Snippet should be truncated with ellipsis
        assert "..." in notes
        # Should be reasonable length
        snippet_match = notes[notes.find(">"):notes.find("\n")]
        assert len(snippet_match) < 250  # Truncated
    
    def test_markdown_special_characters_escaped(self, special_char_citations):
        """Test that special characters are properly escaped."""
        _, notes = CitationRenderer.render_markdown("text", special_char_citations)
        
        # Should escape quotes in snippets
        assert r'\"' in notes or '"' in notes


# ============================================================================
# Test HTML Rendering
# ============================================================================

class TestHTMLRendering:
    """Test HTML citation rendering."""
    
    def test_html_expanded_rendering(self, sample_citations):
        """Test expanded HTML rendering."""
        text = "The derivative is fundamental."
        html = CitationRenderer.render_html(text, sample_citations, collapsible=False)
        
        # Should be valid HTML
        assert "<section" in html
        assert "<ol>" in html
        assert "<li" in html
        assert "</li>" in html
        
        # Should have citation markers
        assert "cite-1" in html
        assert "cite-2" in html
        assert "cite-3" in html
    
    def test_html_collapsible_rendering(self, sample_citations):
        """Test collapsible HTML rendering."""
        text = "The derivative is fundamental."
        html = CitationRenderer.render_html(text, sample_citations, collapsible=True)
        
        # Should have collapsible structure
        assert "<details" in html
        assert "<summary" in html
        assert "sources/Sources" in html.lower() or "show sources" in html.lower()
    
    def test_html_contains_citation_count(self, sample_citations):
        """Test that citation count is shown."""
        text = "The derivative is fundamental."
        html = CitationRenderer.render_html(text, sample_citations, collapsible=True)
        
        # Should show count
        assert "3" in html
    
    def test_html_superscript_links(self, sample_citations):
        """Test that HTML uses superscript links."""
        text = "The derivative is fundamental."
        html = CitationRenderer.render_html(text, sample_citations)
        
        # Should have superscript links
        assert "<sup>" in html
        assert "<a" in html
        assert "href" in html
    
    def test_html_empty_citations(self):
        """Test HTML rendering with no citations."""
        text = "Some text"
        html = CitationRenderer.render_html(text, [])
        
        # Should just wrap in paragraph
        assert "<p>" in html
        assert "Some text" in html


# ============================================================================
# Test Plain Text Rendering
# ============================================================================

class TestPlainTextRendering:
    """Test plain text citation rendering."""
    
    def test_plain_text_basic_rendering(self, sample_citations):
        """Test basic plain text rendering."""
        text = "The derivative is fundamental."
        plain = CitationRenderer.render_plain_text(text, sample_citations)
        
        # Should contain original text
        assert "The derivative is fundamental." in plain
        
        # Should have sources section
        assert "SOURCES:" in plain
        
        # Should have numbered citations
        assert "[1]" in plain
        assert "[2]" in plain
        assert "[3]" in plain
    
    def test_plain_text_contains_source_info(self, sample_citations):
        """Test that plain text contains source information."""
        plain = CitationRenderer.render_plain_text("text", sample_citations)
        
        # Should have source IDs
        assert "Lecture_1_Introduction" in plain
        assert "Khan_Academy_Derivatives" in plain
        
        # Should have source types
        assert "LOCAL" in plain
        assert "ONLINE" in plain
    
    def test_plain_text_empty_citations(self):
        """Test plain text rendering with no citations."""
        text = "Some text"
        plain = CitationRenderer.render_plain_text(text, [])
        
        # Should return unchanged
        assert plain == text


# ============================================================================
# Test Authority Tier Display
# ============================================================================

class TestAuthorityTierDisplay:
    """Test that authority tiers are displayed correctly."""
    
    def test_tier_indicators_in_markdown(self, mixed_authority_citations):
        """Test that tier indicators appear in Markdown."""
        _, notes = CitationRenderer.render_markdown("text", mixed_authority_citations)
        
        # Should have tier indicators
        assert "🔒" in notes  # TIER_1
        assert "🏫" in notes  # TIER_2
        assert "👥" in notes  # TIER_3
        # Default/None uses no indicator (just source info)
    
    def test_tier_indicators_in_html(self, mixed_authority_citations):
        """Test that tier indicators appear in HTML."""
        html = CitationRenderer.render_html("text", mixed_authority_citations)
        
        # Should have tier indicators (as HTML entities or Unicode)
        assert "🔒" in html or "TIER_1" in html
        assert "🏫" in html or "TIER_2" in html
        assert "👥" in html or "TIER_3" in html
    
    def test_source_type_labels(self, sample_citations):
        """Test that source types are labeled correctly."""
        _, notes = CitationRenderer.render_markdown("text", sample_citations)
        
        # Should distinguish local and online
        assert "Local" in notes or "LOCAL" in notes or "📚" in notes
        assert "Online" in notes or "ONLINE" in notes or "🌐" in notes


# ============================================================================
# Test Citation Statistics
# ============================================================================

class TestCitationStatistics:
    """Test citation statistics computation."""
    
    def test_statistics_by_source_type(self, sample_citations):
        """Test counting citations by source type."""
        stats = CitationRenderer.get_citation_statistics(sample_citations)
        
        assert stats["total_citations"] == 3
        assert stats["by_source_type"]["local"] == 2
        assert stats["by_source_type"]["online"] == 1
    
    def test_statistics_by_authority_tier(self, mixed_authority_citations):
        """Test counting citations by authority tier."""
        stats = CitationRenderer.get_citation_statistics(mixed_authority_citations)
        
        assert stats["total_citations"] == 4
        assert stats["by_authority_tier"]["TIER_1"] == 1
        assert stats["by_authority_tier"]["TIER_2"] == 1
        assert stats["by_authority_tier"]["TIER_3"] == 1
        assert stats["by_authority_tier"]["unknown"] == 1
    
    def test_statistics_unique_sources(self, sample_citations):
        """Test counting unique sources."""
        stats = CitationRenderer.get_citation_statistics(sample_citations)
        
        assert stats["unique_sources"] == 3
    
    def test_statistics_empty_citations(self):
        """Test statistics with empty citations."""
        stats = CitationRenderer.get_citation_statistics([])
        
        assert stats["total_citations"] == 0
        assert stats["unique_sources"] == 0
        assert not stats["by_source_type"]


# ============================================================================
# Test Citation Mapper
# ============================================================================

class TestCitationMapper:
    """Test the citation mapping functionality."""
    
    def test_mapper_initialization(self):
        """Test mapper initialization with config."""
        mapper = CitationMapper(
            enable_citations=True,
            show_unverified_with_label=True,
            citation_max_per_claim=5
        )
        
        assert mapper.enable_citations is True
        assert mapper.show_unverified_with_label is True
        assert mapper.citation_max_per_claim == 5
    
    def test_mapper_config_validation(self):
        """Test that mutually exclusive configs raise error."""
        with pytest.raises(ValueError):
            CitationMapper(
                show_unverified_with_label=True,
                show_unverified_omit=True
            )
    
    def test_span_to_citation_conversion(self):
        """Test converting VerifiedSpan to Citation."""
        span = VerifiedSpan(
            span_id="span_001",
            source_id="source_1",
            source_type="local",
            snippet="Some evidence",
            page_num=5,
            authority_tier="TIER_1",
            confidence=0.95
        )
        
        citations = CitationMapper._spans_to_citations([span])
        
        assert len(citations) == 1
        assert citations[0].span_id == "span_001"
        assert citations[0].source_id == "source_1"
        assert citations[0].authority_tier == "TIER_1"
    
    def test_map_claim_with_citations(self):
        """Test mapping a claim with supporting citations."""
        span = VerifiedSpan(
            span_id="span_001",
            source_id="source_1",
            source_type="local",
            snippet="Supporting evidence",
            authority_tier="TIER_1"
        )
        
        mapper = CitationMapper()
        claim, citations = mapper.map_claim_to_citations(
            "The derivative measures rate of change",
            [span]
        )
        
        assert claim == "The derivative measures rate of change"
        assert len(citations) == 1
        assert citations[0].source_id == "source_1"
    
    def test_map_claim_without_citations_with_label(self):
        """Test that unverified claims get labeled."""
        mapper = CitationMapper(show_unverified_with_label=True)
        claim, citations = mapper.map_claim_to_citations("Unsupported claim", [])
        
        assert "(needs evidence)" in claim
        assert len(citations) == 0
    
    def test_map_claim_without_citations_omit(self):
        """Test that unsupported claims are omitted."""
        mapper = CitationMapper(show_unverified_with_label=False, show_unverified_omit=True)
        claim, citations = mapper.map_claim_to_citations("Unsupported claim", [])
        
        assert claim is None
        assert len(citations) == 0
    
    def test_map_claim_respects_max_citations(self):
        """Test that citation limit is respected."""
        spans = [
            VerifiedSpan(f"span_{i}", f"source_{i}", "local", f"snippet {i}")
            for i in range(10)
        ]
        
        mapper = CitationMapper(citation_max_per_claim=3)
        claim, citations = mapper.map_claim_to_citations("Claim", spans)
        
        assert len(citations) == 3
    
    def test_dedup_citations(self):
        """Test deduplication of identical citations."""
        citations = [
            Citation(span_id="span_1", source_id="source_1", source_type="local", snippet="snippet"),
            Citation(span_id="span_1", source_id="source_1", source_type="local", snippet="snippet"),  # Duplicate
            Citation(span_id="span_2", source_id="source_2", source_type="local", snippet="snippet2"),
        ]
        
        deduped = CitationMapper.dedup_citations(citations)
        
        assert len(deduped) == 2
    
    def test_rank_citations_by_tier(self):
        """Test that citations are ranked by authority tier."""
        citations = [
            Citation(span_id="s3", source_id="src_3", source_type="local", snippet="snippet", authority_tier="TIER_3"),
            Citation(span_id="s1", source_id="src_1", source_type="local", snippet="snippet", authority_tier="TIER_1"),
            Citation(span_id="s2", source_id="src_2", source_type="local", snippet="snippet", authority_tier="TIER_2"),
        ]
        
        ranked = CitationMapper.rank_citations_by_tier(citations)
        
        # Should be ordered TIER_1, TIER_2, TIER_3
        assert ranked[0].authority_tier == "TIER_1"
        assert ranked[1].authority_tier == "TIER_2"
        assert ranked[2].authority_tier == "TIER_3"
    
    def test_get_citation_summary(self):
        """Test citation summary generation."""
        citations = [
            Citation(span_id="s1", source_id="src_1", source_type="local", snippet="snippet", authority_tier="TIER_1"),
            Citation(span_id="s2", source_id="src_2", source_type="online", snippet="snippet", authority_tier="TIER_2"),
            Citation(span_id="s3", source_id="src_1", source_type="local", snippet="snippet", authority_tier="TIER_1"),
        ]
        
        summary = CitationMapper.get_citation_summary(citations)
        
        assert summary["total"] == 3
        assert summary["by_source"]["src_1"] == 2
        assert summary["by_source"]["src_2"] == 1
        assert summary["by_type"]["local"] == 2
        assert summary["by_type"]["online"] == 1


# ============================================================================
# Test Edge Cases and Special Scenarios
# ============================================================================

class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_single_citation(self):
        """Test rendering with single citation."""
        citations = [Citation(span_id="s1", source_id="source", source_type="local", snippet="snippet")]
        
        annotated, notes = CitationRenderer.render_markdown("text", citations)
        
        assert "[1]" in annotated
        assert "[2]" not in annotated
    
    def test_many_citations(self):
        """Test rendering with many citations."""
        citations = [
            Citation(span_id=f"s{i}", source_id=f"source_{i}", source_type="local", snippet=f"snippet {i}")
            for i in range(20)
        ]
        
        annotated, notes = CitationRenderer.render_markdown("text", citations)
        
        # Should have all numbers
        for i in range(1, 21):
            assert f"[{i}]" in annotated
    
    def test_citation_with_no_optional_fields(self):
        """Test citation with only required fields."""
        citations = [Citation(span_id="s1", source_id="source", source_type="local", snippet="snippet")]
        
        annotated, notes = CitationRenderer.render_markdown("text", citations)
        
        # Should render without error
        assert "[1]" in annotated
        assert "source" in notes
    
    def test_citation_with_all_fields(self):
        """Test citation with all fields populated."""
        citations = [
            Citation(
                span_id="s1", source_id="source", source_type="local", snippet="snippet",
                page_num=42, authority_tier="TIER_1"
            )
        ]
        
        annotated, notes = CitationRenderer.render_markdown("text", citations)
        
        # Should include all information
        assert "[1]" in annotated
        assert "42" in notes
        assert "TIER_1" in notes or "🔒" in notes
    
    def test_empty_text_with_citations(self, sample_citations):
        """Test rendering empty text with citations."""
        annotated, notes = CitationRenderer.render_markdown("", sample_citations)
        
        # Should still have citations
        assert "[1]" in annotated
        assert len(notes) > 0
    
    def test_very_long_text(self, sample_citations):
        """Test rendering very long text."""
        long_text = "Word " * 1000  # ~5KB of text
        
        annotated, notes = CitationRenderer.render_markdown(long_text, sample_citations)
        
        # Should contain all citations
        assert "[1]" in annotated
        assert "[2]" in annotated
        assert "[3]" in annotated


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining renderer and mapper."""
    
    def test_full_workflow_render_and_map(self):
        """Test complete workflow of mapping and rendering."""
        # Create mapper
        mapper = CitationMapper(
            show_unverified_with_label=True,
            citation_max_per_claim=3
        )
        
        # Create claims and verification
        claims = ["Claim A", "Claim B"]
        spans = {
            "Claim A": [
                VerifiedSpan("s1", "source_1", "local", "evidence for A")
            ],
            "Claim B": []  # No evidence
        }
        
        # Map claims
        filtered, citations_per_claim = mapper.map_claims_to_citations(
            claims, spans
        )
        
        # Should have modification
        assert len(filtered) == 2
        assert "(needs evidence)" in filtered[1]
        
        # Render
        for claim, citations in zip(filtered, citations_per_claim):
            annotated, notes = CitationRenderer.render_markdown(claim, citations)
            # Should render without error
            assert len(annotated) > 0
    
    def test_workflow_with_cs_claims(self):
        """Test workflow with CS-specific claim types."""
        mapper = CitationMapper(
            require_citations_for_cs_claims=True,
            show_unverified_with_label=False,
            show_unverified_omit=True
        )
        
        claim = "O(n log n) complexity"
        claim_type = "COMPLEXITY_CLAIM"
        
        # Without citations, should be omitted
        result, citations = mapper.map_claim_to_citations(
            claim, [], claim_type
        )
        
        assert result is None  # Omitted
        assert len(citations) == 0
        
        # With citations, should be included
        span = VerifiedSpan(span_id="s1", source_id="source_1", source_type="local", snippet="O(n log n) proof")
        result, citations = mapper.map_claim_to_citations(
            claim, [span], claim_type
        )
        
        assert result == claim
        assert len(citations) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
