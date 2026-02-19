"""
Smoke tests for report builder.

Tests that report generation completes successfully for all formats
and includes all required sections.
"""

import pytest
import json
from datetime import datetime
from pathlib import Path
from src.reporting.research_report import (
    ResearchReportBuilder,
    SessionMetadata,
    IngestionReport,
    VerificationSummary,
    ClaimEntry,
    build_report,
)


@pytest.fixture
def sample_session_metadata():
    """Sample session metadata."""
    return SessionMetadata(
        session_id="test_session_001",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        seed=42,
        language_model="gpt-4",
        embedding_model="text-embedding-ada-002",
        nli_model="cross-encoder/qnli",
        inputs_used=["test_source_1.pdf", "test_source_2.txt"],
    )


@pytest.fixture
def sample_ingestion_report():
    """Sample ingestion report."""
    return IngestionReport(
        total_pages=100,
        pages_ocr=95,
        headers_removed=98,
        footers_removed=98,
        watermarks_removed=5,
        chunks_total_all_sources=450,
        avg_chunk_size_all_sources=512,
        extraction_methods=["pdf_text", "pdf_image_ocr", "text_raw"],
    )


@pytest.fixture
def sample_verification_summary():
    """Sample verification summary."""
    return VerificationSummary(
        total_claims=50,
        verified_count=35,
        low_confidence_count=10,
        rejected_count=5,
        avg_confidence=0.82,
        top_rejection_reasons=[
            ("Insufficient evidence", 3),
            ("Contradicted by source", 2),
        ],
        calibration_metrics={
            "ece": 0.0432,
            "brier": 0.1234,
            "log_loss": 0.3421,
        },
    )


@pytest.fixture
def sample_claims():
    """Sample claims list."""
    return [
        ClaimEntry(
            claim_text="The Earth orbits the Sun.",
            status="VERIFIED",
            confidence=0.98,
            evidence_count=5,
            top_evidence="The Earth completes one orbit around the Sun every 365.25 days.",
            page_num=15,
            span_id="para_42",
        ),
        ClaimEntry(
            claim_text="Photosynthesis converts CO2 to glucose.",
            status="VERIFIED",
            confidence=0.95,
            evidence_count=4,
            top_evidence="Plants use photosynthesis to convert carbon dioxide and water into glucose.",
            page_num=28,
            span_id="para_87",
        ),
        ClaimEntry(
            claim_text="Water boils at exactly 100 degrees Celsius at all altitudes.",
            status="LOW_CONFIDENCE",
            confidence=0.65,
            evidence_count=2,
            top_evidence="Water boils at 100°C at sea level, but at lower temperatures at higher altitudes.",
            page_num=42,
            span_id="para_156",
        ),
        ClaimEntry(
            claim_text="Some unverifiable claim about alien technology.",
            status="REJECTED",
            confidence=0.15,
            evidence_count=0,
            top_evidence="No evidence found in source materials.",
            page_num=None,
            span_id=None,
        ),
    ]


@pytest.fixture
def sample_performance_metrics():
    """Sample performance metrics."""
    return {
        "total_inference_time_sec": 45.3,
        "avg_claim_extraction_time_ms": 120.5,
        "avg_verification_time_ms": 456.2,
        "memory_peak_mb": 2048,
        "evidence_retrieval_count": 285,
    }


class TestReportBuilderMarkdown:
    """Test Markdown report generation."""

    def test_markdown_generation_succeeds(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that Markdown report generates without errors."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        md = builder.build_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_markdown_contains_title(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that Markdown includes title."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        md = builder.build_markdown()
        assert "# AI Verification Session Report" in md

    def test_markdown_contains_session_info(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that Markdown includes session information."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        md = builder.build_markdown()
        assert "## Session Information" in md
        assert sample_session_metadata.session_id in md
        assert sample_session_metadata.version in md

    def test_markdown_contains_ingestion_stats(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that Markdown includes ingestion statistics."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        md = builder.build_markdown()
        assert "## Ingestion Statistics" in md
        assert "100" in md  # total_pages
        assert "450" in md  # total_chunks

    def test_markdown_contains_verification_results(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that Markdown includes verification statistics."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        md = builder.build_markdown()
        assert "## Verification Results" in md
        assert "Verified" in md
        assert "Low Confidence" in md
        assert "Rejected" in md

    def test_markdown_contains_trust_guidance(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that Markdown includes trust guidance."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        md = builder.build_markdown()
        assert "What to Trust / What Not to Trust" in md
        assert "You Can Trust Claims That Are" in md
        assert "Do Not Trust" in md

    def test_markdown_contains_claim_table(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that Markdown includes claim table."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        md = builder.build_markdown()
        assert "## Verified Claims Table" in md
        assert "Verified Claims" in md
        assert "Claim" in md

    def test_markdown_with_performance_metrics(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
        sample_performance_metrics,
    ):
        """Test that Markdown includes performance metrics when provided."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)
        builder.add_performance_metrics(sample_performance_metrics)

        md = builder.build_markdown()
        assert "## Performance Metrics" in md
        assert "inference_time" in md


class TestReportBuilderHTML:
    """Test HTML report generation."""

    def test_html_generation_succeeds(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that HTML report generates without errors."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        html = builder.build_html()
        assert isinstance(html, str)
        assert len(html) > 0

    def test_html_is_valid_markup(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that HTML is valid."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        html = builder.build_html()
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_html_contains_styles(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that HTML includes styling."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        html = builder.build_html()
        assert "<style>" in html
        assert "</style>" in html
        assert "font-family" in html

    def test_html_contains_content(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that HTML contains report content."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        html = builder.build_html()
        assert "AI Verification Session Report" in html
        assert "Session Information" in html


class TestReportBuilderJSON:
    """Test JSON audit report generation."""

    def test_json_generation_succeeds(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that JSON audit report generates without errors."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        audit = builder.build_audit_json()
        assert isinstance(audit, dict)

    def test_json_serializable(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that JSON can be serialized to string."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        audit = builder.build_audit_json()
        json_str = json.dumps(audit)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_json_contains_required_fields(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that JSON contains all required top-level fields."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        audit = builder.build_audit_json()
        assert "report_type" in audit
        assert "generated_at" in audit
        assert "session" in audit
        assert "ingestion" in audit
        assert "verification" in audit
        assert "claims" in audit
        assert "metadata" in audit

    def test_json_metadata_structure(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test JSON metadata structure."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        audit = builder.build_audit_json()
        assert audit["report_type"] == "audit"
        assert "generated_at" in audit
        assert audit["metadata"]["report_version"] == "1.0"

    def test_json_claims_structure(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test JSON claims array structure."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)

        audit = builder.build_audit_json()
        assert isinstance(audit["claims"], list)
        assert len(audit["claims"]) == len(sample_claims)
        assert "claim_text" in audit["claims"][0]
        assert "status" in audit["claims"][0]
        assert "confidence" in audit["claims"][0]


class TestBuildReportConvenienceFunction:
    """Test the convenience function build_report()."""

    def test_build_report_returns_three_formats(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that build_report returns tuple of (md, html, json)."""
        md, html, audit_json = build_report(
            sample_session_metadata,
            sample_ingestion_report,
            sample_verification_summary,
            sample_claims,
        )

        assert isinstance(md, str)
        assert isinstance(html, str)
        assert isinstance(audit_json, dict)

    def test_build_report_with_performance_metrics(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
        sample_performance_metrics,
    ):
        """Test that build_report handles performance metrics."""
        md, html, audit_json = build_report(
            sample_session_metadata,
            sample_ingestion_report,
            sample_verification_summary,
            sample_claims,
            sample_performance_metrics,
        )

        assert "Performance Metrics" in md
        assert audit_json["performance"] == sample_performance_metrics

    def test_build_report_all_formats_have_content(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that all formats have meaningful content."""
        md, html, audit_json = build_report(
            sample_session_metadata,
            sample_ingestion_report,
            sample_verification_summary,
            sample_claims,
        )

        # Markdown checks
        assert len(md) > 500
        assert "Session Information" in md

        # HTML checks
        assert len(html) > 1000
        assert "<html>" in html
        assert "Session Information" in html

        # JSON checks
        assert len(audit_json["claims"]) == len(sample_claims)
        assert audit_json["session"]["session_id"] == sample_session_metadata.session_id


class TestBuilderChaining:
    """Test builder method chaining."""

    def test_builder_methods_return_self(self, sample_session_metadata):
        """Test that builder methods return self for chaining."""
        builder = ResearchReportBuilder()
        
        result = builder.add_session_metadata(sample_session_metadata)
        assert result is builder

    def test_complete_chain(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
        sample_performance_metrics,
    ):
        """Test complete method chaining."""
        md = (
            ResearchReportBuilder()
            .add_session_metadata(sample_session_metadata)
            .add_ingestion_report(sample_ingestion_report)
            .add_verification_summary(sample_verification_summary)
            .add_claims(sample_claims)
            .add_performance_metrics(sample_performance_metrics)
            .build_markdown()
        )

        assert isinstance(md, str)
        assert "AI Verification Session Report" in md
        assert "Performance Metrics" in md


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_claims_list(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
    ):
        """Test report generation with no claims."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims([])

        md = builder.build_markdown()
        # When there are no claims, the claims table section is not rendered
        # but the report should still be valid with all other sections
        assert "Session Information" in md
        assert "Ingestion Statistics" in md
        assert "Verification Results" in md

    def test_single_claim(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test report generation with single claim."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims([sample_claims[0]])

        md = builder.build_markdown()
        assert "Verified Claims" in md
        assert sample_claims[0].claim_text[:50] in md

    def test_no_optional_metrics(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
        sample_claims,
    ):
        """Test that report works without optional performance metrics."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims(sample_claims)
        # Don't add performance metrics

        md, html, audit = builder.build_report()

        assert isinstance(md, str)
        assert isinstance(html, str)
        assert isinstance(audit, dict)
        # Performance should be empty dict
        assert audit["performance"] == {}

    def test_special_characters_in_claim_text(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
    ):
        """Test handling of special characters in claim text."""
        special_claim = ClaimEntry(
            claim_text="The equation E=mc² shows mass-energy equivalence | special & chars",
            status="VERIFIED",
            confidence=0.9,
            evidence_count=3,
            top_evidence="Einstein's theory of relativity.",
            page_num=10,
        )

        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims([special_claim])

        md = builder.build_markdown()
        html = builder.build_html()

        assert len(md) > 0
        assert len(html) > 0

    def test_very_long_claim_text(
        self,
        sample_session_metadata,
        sample_ingestion_report,
        sample_verification_summary,
    ):
        """Test handling of very long claim text."""
        long_claim = ClaimEntry(
            claim_text="A" * 1000,  # 1000 character claim
            status="VERIFIED",
            confidence=0.85,
            evidence_count=2,
            top_evidence="Some evidence.",
            page_num=5,
        )

        builder = ResearchReportBuilder()
        builder.add_session_metadata(sample_session_metadata)
        builder.add_ingestion_report(sample_ingestion_report)
        builder.add_verification_summary(sample_verification_summary)
        builder.add_claims([long_claim])

        md = builder.build_markdown()
        assert len(md) > 0
        # Should be truncated in table
        assert "..." in md
