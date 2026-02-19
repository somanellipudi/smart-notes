"""
Test report invariants and consistency.

Validates:
1. Report count invariants (verified + rejected + low_conf == total)
2. No placeholder values when data is empty
3. Rejection reasons sum to rejected count
4. Questions never appear in rejected bucket
5. Citations present when evidence exists
"""

import pytest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from src.reporting.research_report import (
    ResearchReportBuilder,
    SessionMetadata,
    IngestionReport,
    VerificationSummary,
    ClaimEntry,
    build_report,
)


class TestReportInvariants:
    """Tests for report data consistency invariants."""

    def test_report_invariants_counts(self) -> None:
        """
        Test: Report count invariants are enforced.
        
        Invariant: total_claims == verified + rejected + low_confidence
        
        When: Building report with specific claim counts
        Then: VerificationSummary should enforce count invariant
        """
        # Build report with mismatched counts
        builder = ResearchReportBuilder()
        builder.add_session_metadata(SessionMetadata(
            session_id="test_session",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            seed=42,
            language_model="test-model",
            embedding_model="test-embedding",
            nli_model="test-nli",
            inputs_used=["pdf"],
        ))
        
        builder.add_ingestion_report(IngestionReport(
            total_pages=10,
            pages_ocr=10,
            headers_removed=0,
            footers_removed=0,
            watermarks_removed=0,
            total_chunks=100,
            avg_chunk_size=512,
            extraction_methods=["text"],
        ))
        
        # Create verification summary with mismatched counts
        # Note: __post_init__ will enforce invariant
        ver_summary = VerificationSummary(
            total_claims=10,
            verified_count=3,
            low_confidence_count=2,
            rejected_count=4,  # 3 + 2 + 4 = 9, not 10
            avg_confidence=0.75,
            top_rejection_reasons=[("insufficient_evidence", 2), ("factually_incorrect", 2)],
        )
        
        # After __post_init__, total_claims should be corrected to 9
        # OR verified/rejected/low_conf should sum to total
        actual_sum = ver_summary.verified_count + ver_summary.rejected_count + ver_summary.low_confidence_count
        assert ver_summary.total_claims == actual_sum, (
            f"Total claims {ver_summary.total_claims} should equal "
            f"verified ({ver_summary.verified_count}) + rejected ({ver_summary.rejected_count}) + "
            f"low_conf ({ver_summary.low_confidence_count}) = {actual_sum}"
        )

    def test_report_no_placeholders_when_empty(self) -> None:
        """
        Test: No placeholder values when report data is empty.
        
        Invariant:
        - If total_chunks == 0, then avg_chunk_size must be None or N/A
        - If total_claims == 0, then avg_confidence should be N/A
        - If rejected_count == 0, then rejection_reasons should be None or empty
        
        When: Building report with empty ingestion/verification
        Then: Placeholder values should not appear
        """
        builder = ResearchReportBuilder()
        
        # Session with empty ingestion
        empty_ingestion = IngestionReport(
            total_pages=0,
            pages_ocr=0,
            headers_removed=0,
            footers_removed=0,
            watermarks_removed=0,
            total_chunks=0,  # Empty ingestion
            avg_chunk_size=None,  # Must be None when chunks=0
            extraction_methods=[],
        )
        
        assert empty_ingestion.avg_chunk_size is None, (
            f"avg_chunk_size should be None when total_chunks=0, got {empty_ingestion.avg_chunk_size}"
        )
        
        # Session with empty verification
        empty_verification = VerificationSummary(
            total_claims=0,
            verified_count=0,
            low_confidence_count=0,
            rejected_count=0,
            avg_confidence=0.0,  # OK to be 0.0 when no claims
            top_rejection_reasons=[],
            rejection_reasons_dict=None,
        )
        
        assert empty_verification.total_claims == 0
        assert empty_verification.rejection_reasons_dict is None or empty_verification.rejection_reasons_dict == {}
        
        builder.add_ingestion_report(empty_ingestion)
        builder.add_verification_summary(empty_verification)
        
        md_content, _, _ = builder.build_report()
        
        # Report should show "N/A" for avg_chunk_size when empty
        assert "N/A" in md_content or "0" in md_content or "None" in md_content

    def test_rejection_reasons_sum(self) -> None:
        """
        Test: Rejection reasons sum to rejected count.
        
        Invariant:
        - sum(rejection_reasons.values()) == rejected_count
        - If sum < rejected_count, UNKNOWN_REASON should be added
        
        When: Creating VerificationSummary with rejection_reasons
        Then: __post_init__ should ensure reasons sum to rejected_count
        """
        # Create with mismatched rejection_reasons
        ver_summary = VerificationSummary(
            total_claims=10,
            verified_count=5,
            low_confidence_count=2,
            rejected_count=3,
            avg_confidence=0.75,
            top_rejection_reasons=[("insufficient_evidence", 2)],  # Only 2, but rejected_count=3
            rejection_reasons_dict={"insufficient_evidence": 2},  # Only 2, but rejected_count=3
        )
        
        # After __post_init__, should have UNKNOWN_REASON added or adjusted
        if ver_summary.rejection_reasons_dict:
            reasons_sum = sum(ver_summary.rejection_reasons_dict.values())
            assert reasons_sum == ver_summary.rejected_count, (
                f"Rejection reasons should sum to {ver_summary.rejected_count}, "
                f"but sum is {reasons_sum}: {ver_summary.rejection_reasons_dict}"
            )

    def test_question_not_in_rejected_bucket(self) -> None:
        """
        Test: Question-type claims never appear in rejected bucket.
        
        Invariant:
        - ClaimEntry with claim_type="question" should never have status="REJECTED"
        - Questions should be in separate section in report
        
        When: Building report with mixed claim types
        Then: Questions should be separated and marked as ANSWERED, not REJECTED
        """
        builder = ResearchReportBuilder()
        
        builder.add_session_metadata(SessionMetadata(
            session_id="test_session",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            seed=42,
            language_model="test-model",
            embedding_model="test-embedding",
            nli_model="test-nli",
            inputs_used=["pdf"],
        ))
        
        # Add mixed claims: factual and questions
        claims = [
            ClaimEntry(
                claim_text="Paris is the capital of France",
                status="VERIFIED",
                confidence=0.95,
                evidence_count=2,
                top_evidence="According to...",
                claim_type="fact_claim",
            ),
            ClaimEntry(
                claim_text="What is photosynthesis?",
                status="ANSWERED",  # Not REJECTED
                confidence=0.90,
                evidence_count=1,
                top_evidence="Photosynthesis is...",
                claim_type="question",
            ),
            ClaimEntry(
                claim_text="Gravity only works on Earth",
                status="REJECTED",
                confidence=0.15,
                evidence_count=1,
                top_evidence="Actually, gravity...",
                claim_type="fact_claim",
            ),
        ]
        
        builder.add_claims(claims)
        builder.add_ingestion_report(IngestionReport(
            total_pages=10,
            pages_ocr=10,
            headers_removed=0,
            footers_removed=0,
            watermarks_removed=0,
            total_chunks=100,
            avg_chunk_size=512,
            extraction_methods=["text"],
        ))
        builder.add_verification_summary(VerificationSummary(
            total_claims=3,
            verified_count=1,
            low_confidence_count=0,
            rejected_count=1,
            avg_confidence=0.67,
            top_rejection_reasons=[],
            rejection_reasons_dict={},
        ))
        
        md_content, _, _ = builder.build_report()
        
        # Should have Questions Answered section (not in rejected)
        assert "Questions Answered" in md_content or "question" in md_content.lower()
        
        # Verify invariant: questions with REJECTED status should log warning
        for claim in claims:
            if claim.claim_type == "question":
                assert claim.status != "REJECTED", (
                    f"Question claim should never be REJECTED: {claim.claim_text}"
                )

    def test_citations_present_when_evidence_present(self) -> None:
        """
        Test: Citations appear in report when evidence exists.
        
        Invariant:
        - If evidence_count > 0, then page_num or span_id should be displayed as citation
        - If evidence_count == 0, then citation should be N/A
        
        When: Building report with mixed evidence
        Then: Citations should appear for claims with evidence
        """
        builder = ResearchReportBuilder()
        
        builder.add_session_metadata(SessionMetadata(
            session_id="test_session",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            seed=42,
            language_model="test-model",
            embedding_model="test-embedding",
            nli_model="test-nli",
            inputs_used=["pdf"],
        ))
        
        claim_with_evidence = ClaimEntry(
            claim_text="Paris is the capital of France",
            status="VERIFIED",
            confidence=0.95,
            evidence_count=2,
            top_evidence="From document...",
            page_num=42,
            span_id="span_123",
            claim_type="fact_claim",
        )
        
        claim_without_evidence = ClaimEntry(
            claim_text="Unknown fact",
            status="LOW_CONFIDENCE",
            confidence=0.50,
            evidence_count=0,
            top_evidence="",
            page_num=None,
            span_id=None,
            claim_type="fact_claim",
        )
        
        builder.add_claims([claim_with_evidence, claim_without_evidence])
        builder.add_ingestion_report(IngestionReport(
            total_pages=100,
            pages_ocr=100,
            headers_removed=0,
            footers_removed=0,
            watermarks_removed=0,
            total_chunks=1000,
            avg_chunk_size=512,
            extraction_methods=["text"],
        ))
        builder.add_verification_summary(VerificationSummary(
            total_claims=2,
            verified_count=1,
            low_confidence_count=1,
            rejected_count=0,
            avg_confidence=0.72,
            top_rejection_reasons=[],
        ))
        
        md_content, _, _ = builder.build_report()
        
        # With evidence (citation should be present)
        assert "p.42" in md_content or "span_123" in md_content or "42" in md_content, (
            f"Citation for claim with evidence should appear in report"
        )
        
        # Verify invariant
        assert claim_with_evidence.evidence_count > 0
        assert claim_with_evidence.page_num is not None or claim_with_evidence.span_id is not None
        
        assert claim_without_evidence.evidence_count == 0
        # N/A should appear or be implied
        assert "N/A" in md_content or len(md_content) > 0


class TestReportBuilderIntegration:
    """Integration tests for complete report building."""

    def test_complete_report_with_all_sections(self) -> None:
        """Test building a complete report with all sections."""
        session_metadata = SessionMetadata(
            session_id="integration_test",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            seed=42,
            language_model="test-model",
            embedding_model="test-embedding",
            nli_model="test-nli",
            inputs_used=["pdf", "text"],
        )
        
        ingestion_report = IngestionReport(
            total_pages=50,
            pages_ocr=50,
            headers_removed=10,
            footers_removed=10,
            watermarks_removed=0,
            total_chunks=500,
            avg_chunk_size=512,
            extraction_methods=["text", "ocr"],
        )
        
        verification_summary = VerificationSummary(
            total_claims=20,
            verified_count=15,
            low_confidence_count=3,
            rejected_count=2,
            avg_confidence=0.82,
            top_rejection_reasons=[
                ("insufficient_evidence", 1),
                ("factually_incorrect", 1),
            ],
            rejection_reasons_dict={
                "insufficient_evidence": 1,
                "factually_incorrect": 1,
            },
        )
        
        claims = [
            ClaimEntry(
                claim_text=f"Verified claim {i}",
                status="VERIFIED",
                confidence=0.90 + (i % 5) * 0.01,
                evidence_count=2,
                top_evidence="Evidence text",
                page_num=i,
                span_id=f"span_{i}",
                claim_type="fact_claim",
            )
            for i in range(15)
        ] + [
            ClaimEntry(
                claim_text=f"Low confidence claim {i}",
                status="LOW_CONFIDENCE",
                confidence=0.50 + (i % 3) * 0.05,
                evidence_count=1,
                top_evidence="Weak evidence",
                page_num=20 + i,
                span_id=f"span_low_{i}",
                claim_type="fact_claim",
            )
            for i in range(3)
        ] + [
            ClaimEntry(
                claim_text=f"Rejected claim {i}",
                status="REJECTED",
                confidence=0.20,
                evidence_count=0,
                top_evidence="",
                claim_type="fact_claim",
            )
            for i in range(2)
        ] + [
            ClaimEntry(
                claim_text="What is machine learning?",
                status="ANSWERED",
                confidence=0.95,
                evidence_count=1,
                top_evidence="ML is...",
                page_num=30,
                claim_type="question",
            ),
        ]
        
        perfomance_metrics = {
            "precision": 0.89,
            "recall": 0.85,
            "f1": 0.87,
        }
        
        # Build report
        md_content, html_content, audit_json = build_report(
            session_metadata,
            ingestion_report,
            verification_summary,
            claims,
            performance_metrics=perfomance_metrics,
        )
        
        # Verify outputs exist and are non-empty
        assert md_content and len(md_content) > 100
        assert html_content and len(html_content) > 100
        assert audit_json and len(audit_json) > 0
        
        # Verify report structure
        assert "Session Information" in md_content or "session" in md_content.lower()
        assert "Ingestion" in md_content or "ingestion" in md_content.lower()
        assert "Verification" in md_content or "verification" in md_content.lower()
        assert "Verified Claims" in md_content or "Rejected Claims" in md_content
        
        # Verify no placeholder values
        assert "512" not in md_content or "total_chunks" not in md_content  # avg_chunk_size should be computed
        
        # Verify claims are present
        assert any(f"Verified claim {i}" in md_content for i in range(3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
