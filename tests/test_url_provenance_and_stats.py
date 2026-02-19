"""
Tests for URL provenance in citations and multi-source ingestion statistics.

Validates that:
1. Citations include URL provenance when source is URL
2. Extraction counts include all sources (PDF + text + URL + audio)
3. Invariants hold (claims > 0 implies chunks > 0, evidence > 0 implies citations exist)
4. Reports show URL counts when URLs are present
"""

import pytest
from src.retrieval.semantic_retriever import EvidenceSpan
from src.retrieval.evidence_store import Evidence
from src.schema.verifiable_schema import EvidenceCitation
from src.reporting.research_report import IngestionReport, ClaimEntry
from src.reporting.run_context import IngestionReportContext
from src.reporting.ingestion_stats import IngestionStatsAggregator


class TestCitationURLProvenance:
    """Test that citations include URL provenance."""
    
    def test_evidence_span_has_url_fields(self):
        """Test EvidenceSpan includes URL provenance fields."""
        span = EvidenceSpan(
            text="Test evidence from YouTube",
            source_type="youtube_transcript",
            source_id="yt_123",
            span_start=0,
            span_end=100,
            similarity=0.95,
            origin="https://youtube.com/watch?v=abc123",
            timestamp_range=(65.0, 95.0)
        )
        
        assert span.origin == "https://youtube.com/watch?v=abc123"
        assert span.timestamp_range == (65.0, 95.0)
        assert span.source_type == "youtube_transcript"
    
    def test_evidence_citation_has_provenance_fields(self):
        """Test EvidenceCitation includes all provenance fields."""
        citation = EvidenceCitation(
            source_type="url_article",
            quote="Python is a high-level programming language",
            location="paragraph 2",
            origin="https://example.com/python-intro",
            confidence=0.92
        )
        
        assert citation.origin == "https://example.com/python-intro"
        assert citation.source_type == "url_article"
    
    def test_pdf_citation_has_page_num(self):
        """Test PDF citations include page numbers."""
        citation = EvidenceCitation(
            source_type="pdf_page",
            quote="Machine learning is a subset of AI",
            origin="textbook.pdf",
            page_num=42,
            confidence=0.98
        )
        
        assert citation.page_num == 42
        assert citation.origin == "textbook.pdf"
    
    def test_video_citation_has_timestamp(self):
        """Test video citations include timestamp range."""
        citation = EvidenceCitation(
            source_type="youtube_transcript",
            quote="This algorithm has O(n log n) complexity",
            origin="https://youtube.com/watch?v=xyz",
            timestamp_range=(125.5, 142.0),
            confidence=0.88
        )
        
        assert citation.timestamp_range == (125.5, 142.0)
        assert citation.origin.startswith("https://youtube.com")


class TestMultiSourceIngestionStats:
    """Test that ingestion statistics track all sources."""
    
    def test_ingestion_report_has_url_fields(self):
        """Test IngestionReport includes URL-specific fields."""
        report = IngestionReport(
            total_pages=0,
            pages_ocr=0,
            headers_removed=0,
            footers_removed=0,
            watermarks_removed=0,
            url_count=5,
            url_fetch_success_count=4,
            url_chunks_total=120,
            text_chars_total=5000,
            text_chunks_total=15,
            audio_seconds=180.0,
            transcript_chars=8000,
            transcript_chunks_total=25,
            chunks_total_all_sources=160,
            extraction_methods=["url", "text", "audio"]
        )
        
        assert report.url_count == 5
        assert report.url_fetch_success_count == 4
        assert report.url_chunks_total == 120
        assert report.text_chunks_total == 15
        assert report.transcript_chunks_total == 25
        assert report.chunks_total_all_sources == 160
    
    def test_ingestion_context_validates_url_invariants(self):
        """Test that IngestionReportContext validates URL invariants."""
        context = IngestionReportContext(
            url_count=10,
            url_fetch_success_count=12,  # Violates invariant
            chunks_total_all_sources=100
        )
        
        violations = context.validate_invariants()
        assert len(violations) > 0
        assert any("url_fetch_success_count" in v.lower() for v in violations)
    
    def test_stats_aggregator_tracks_multiple_sources(self):
        """Test IngestionStatsAggregator tracks all source types."""
        aggregator = IngestionStatsAggregator()
        
        # Add PDF
        aggregator.add_pdf_source(
            source_id="doc.pdf",
            pages=10,
            pages_ocr=2,
            chunks=30,
            chars=15000
        )
        
        # Add URLs
        aggregator.add_url_source(
            source_id="https://example.com/1",
            fetch_success=True,
            chunks=20,
            chars=10000
        )
        aggregator.add_url_source(
            source_id="https://example.com/2",
            fetch_success=False
        )
        
        # Add text
        aggregator.add_text_source(
            source_id="notes",
            chunks=10,
            chars=5000
        )
        
        # Add audio
        aggregator.add_audio_source(
            source_id="lecture.mp3",
            duration_seconds=1800.0,
            transcript_chars=12000,
            chunks=40
        )
        
        # Validate
        assert aggregator.pdf_pages == 10
        assert aggregator.url_count == 2
        assert aggregator.url_fetch_success == 1
        assert aggregator.text_chunks == 10
        assert aggregator.transcript_chunks == 40
        assert aggregator.get_total_chunks() == 100  # 30 + 20 + 10 + 40
        assert aggregator.get_total_chars() == 42000  # 15000 + 10000 + 5000 + 12000
    
    def test_stats_aggregator_computes_avg_chunk_size(self):
        """Test average chunk size computation."""
        aggregator = IngestionStatsAggregator()
        
        aggregator.add_text_source("text", chunks=10, chars=5000)
        aggregator.add_url_source("url", fetch_success=True, chunks=10, chars=3000)
        
        avg_size = aggregator.get_avg_chunk_size()
        assert avg_size == 400.0  # 8000 / 20
    
    def test_stats_aggregator_returns_none_for_zero_chunks(self):
        """Test that avg_chunk_size is None when no chunks."""
        aggregator = IngestionStatsAggregator()
        
        assert aggregator.get_avg_chunk_size() is None
        assert aggregator.get_total_chunks() == 0


class TestIngestionInvariants:
    """Test ingestion statistics invariants."""
    
    def test_invariant_claims_imply_chunks(self):
        """Test: if total_claims > 0 then chunks_total_all_sources must be > 0."""
        # This would be tested at the pipeline level where claims are generated
        # from chunks. Here we test the stats side.
        
        aggregator = IngestionStatsAggregator()
        # No sources added
        
        # Should be safe to have 0 chunks with 0 claims
        assert aggregator.get_total_chunks() == 0
        
        # Add some chunks
        aggregator.add_text_source("notes", chunks=5, chars=2000)
        assert aggregator.get_total_chunks() > 0
    
    def test_invariant_evidence_implies_citations(self):
        """Test: if evidence_count > 0, citations must be non-empty."""
        claim = ClaimEntry(
            claim_text="Test claim",
            status="VERIFIED",
            confidence=0.95,
            evidence_count=2,
            top_evidence="Evidence snippet",
            citation_origin="https://example.com",
            citation_source_type="url_article"
        )
        
        # Should have citation info when evidence_count > 0
        assert claim.evidence_count > 0
        assert claim.citation_origin is not None
    
    def test_chunk_size_none_when_no_chunks(self):
        """Test avg_chunk_size is None when chunks_total_all_sources == 0."""
        report = IngestionReport(
            total_pages=0,
            pages_ocr=0,
            headers_removed=0,
            footers_removed=0,
            watermarks_removed=0,
            chunks_total_all_sources=0,
            avg_chunk_size_all_sources=100.0,  # Should be corrected to None
            extraction_methods=[]
        )
        
        # __post_init__ should set this to None
        assert report.avg_chunk_size_all_sources is None


class TestReportURLDisplay:
    """Test that reports show URL information correctly."""
    
    def test_claim_entry_has_url_fields(self):
        """Test ClaimEntry includes URL citation fields."""
        claim = ClaimEntry(
            claim_text="Python supports multiple programming paradigms",
            status="VERIFIED",
            confidence=0.92,
            evidence_count=3,
            top_evidence="Python is a versatile language",
            citation_origin="https://docs.python.org/3/tutorial",
            citation_source_type="url_article",
            citation_timestamp=None
        )
        
        assert claim.citation_origin == "https://docs.python.org/3/tutorial"
        assert claim.citation_source_type == "url_article"
    
    def test_youtube_citation_has_timestamp(self):
        """Test YouTube citations include timestamp."""
        claim = ClaimEntry(
            claim_text="Insertion sort is O(n²) in worst case",
            status="VERIFIED",
            confidence=0.88,
            evidence_count=1,
            top_evidence="Worst case analysis shows...",
            citation_origin="https://youtube.com/watch?v=abc",
            citation_source_type="youtube_transcript",
            citation_timestamp="02:15-02:45"
        )
        
        assert claim.citation_timestamp == "02:15-02:45"
        assert "youtube.com" in claim.citation_origin


class TestStatsValidation:
    """Test validation of ingestion statistics."""
    
    def test_validate_url_success_rate(self):
        """Test URL success rate validation."""
        aggregator = IngestionStatsAggregator()
        
        aggregator.add_url_source("url1", fetch_success=True, chunks=10, chars=1000)
        aggregator.add_url_source("url2", fetch_success=True, chunks=15, chars=1500)
        aggregator.add_url_source("url3", fetch_success=False)
        
        assert aggregator.url_count == 3
        assert aggregator.url_fetch_success == 2
        
        errors = aggregator.validate()
        assert len(errors) == 0
    
    def test_validate_pdf_ocr_pages(self):
        """Test PDF OCR pages validation."""
        aggregator = IngestionStatsAggregator()
        
        aggregator.add_pdf_source(
            "doc.pdf",
            pages=10,
            pages_ocr=12,  # Invalid: exceeds total pages
            chunks=20,
            chars=10000
        )
        
        errors = aggregator.validate()
        assert len(errors) > 0
        assert any("ocr" in e.lower() for e in errors)
    
    def test_to_ingestion_report_conversion(self):
        """Test conversion to IngestionReport."""
        aggregator = IngestionStatsAggregator()
        
        aggregator.add_url_source("url", fetch_success=True, chunks=20, chars=8000)
        aggregator.add_text_source("text", chunks=10, chars=4000)
        
        report = aggregator.to_ingestion_report()
        
        assert isinstance(report, IngestionReport)
        assert report.url_count == 1
        assert report.url_fetch_success_count == 1
        assert report.text_chunks_total == 10
        assert report.chunks_total_all_sources == 30
        assert report.avg_chunk_size_all_sources == 400.0


class TestSourceTypeHandling:
    """Test handling of different source types."""
    
    def test_evidence_span_source_types(self):
        """Test all supported source types in EvidenceSpan."""
        source_types = [
            "pdf_page",
            "notes_text",
            "external_context",
            "url_article",
            "youtube_transcript",
            "audio_transcript"
        ]
        
        for source_type in source_types:
            span = EvidenceSpan(
                text="Test",
                source_type=source_type,
                source_id="test_id",
                span_start=0,
                span_end=50,
                similarity=0.9
            )
            assert span.source_type == source_type
    
    def test_evidence_citation_source_types(self):
        """Test all supported source types in EvidenceCitation."""
        source_types = [
            "pdf_page",
            "notes_text",
            "url_article",
            "youtube_transcript",
            "audio_transcript",
            "external_context",
            "equation",
            "inferred"
        ]
        
        for source_type in source_types:
            citation = EvidenceCitation(
                source_type=source_type,
                quote="Test quote from source"
            )
            assert citation.source_type == source_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
