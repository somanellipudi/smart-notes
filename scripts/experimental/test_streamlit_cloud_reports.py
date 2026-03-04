"""
Test to verify report generation works on Streamlit Cloud environment with OCR disabled.
This simulates the user-reported issue.
"""

import os
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Set Streamlit Cloud environment
os.environ["STREAMLIT_CLOUD"] = "true"

# Now import config to verify OCR is disabled
import config
print(f"Environment: {'Streamlit Cloud' if config.IS_STREAMLIT_CLOUD else 'Local'}")
print(f"OCR_ENABLED: {config.OCR_ENABLED}")
print(f"ENABLE_OCR_FALLBACK: {config.ENABLE_OCR_FALLBACK}")

from src.reporting.research_report import build_report, SessionMetadata, IngestionReport, VerificationSummary, ClaimEntry

def test_report_with_manual_text_ingestion():
    """Test report generation with text-only input (no OCR/PDF)."""
    
    print("\n" + "=" * 60)
    print("TEST: Report generation with manual text input")
    print("(Simulating Streamlit Cloud with OCR disabled)")
    print("=" * 60)
    
    # Create session metadata
    session_metadata = SessionMetadata(
        session_id="test-session-cloud",
        timestamp="2026-02-21T14:00:00",
        version="1.0.0",
        seed=42,
        language_model="gpt-4",
        embedding_model="text-embedding-ada-002",
        nli_model="cross-encoder/qnli",
        inputs_used=["manual_text"],
    )
    
    # Create MINIMAL ingestion report (OCR not available on Cloud)
    ingestion_report = IngestionReport(
        total_pages=0,
        pages_ocr=0,
        headers_removed=0,
        footers_removed=0,
        watermarks_removed=0,
        # No URL/Audio/OCR data since OCR disabled
        url_count=0,
        url_fetch_success_count=0,
        url_chunks_total=0,
        text_chars_total=500,  # Manual text input
        text_chunks_total=0,
        audio_seconds=0.0,
        transcript_chars=0,
        transcript_chunks_total=0,
        chunks_total_all_sources=0,
        avg_chunk_size_all_sources=None,
        extraction_methods=["manual_text"],  # Only source
    )
    
    print("\n✓ Created IngestionReport with:")
    print(f"  - Extraction methods: {ingestion_report.extraction_methods}")
    print(f"  - Text chars: {ingestion_report.text_chars_total}")
    print(f"  - Total pages (PDF): {ingestion_report.total_pages}")
    
    # Create verification summary
    verification_summary = VerificationSummary(
        total_claims=5,
        verified_count=3,
        low_confidence_count=1,
        rejected_count=1,
        avg_confidence=0.75,
        top_rejection_reasons=[("insufficient_evidence", 1)],
    )
    
    print("\n✓ Created VerificationSummary with:")
    print(f"  - Total claims: {verification_summary.total_claims}")
    print(f"  - Verified: {verification_summary.verified_count}")
    
    # Create sample claims
    claims_entries = [
        ClaimEntry(
            claim_text="Sample claim 1",
            status="VERIFIED",
            confidence=0.85,
            evidence_count=2,
            top_evidence="Supporting evidence",
            claim_type="fact_claim",
        ),
        ClaimEntry(
            claim_text="Sample claim 2",
            status="UNVERIFIABLE",
            confidence=0.5,
            evidence_count=0,
            top_evidence="",
            claim_type="fact_claim",
        ),
    ]
    
    print(f"\n✓ Created {len(claims_entries)} claim entries")
    
    # Build report
    print("\n⏳ Building research reports...")
    try:
        md_content, html_content, audit_json = build_report(
            session_metadata,
            ingestion_report,
            verification_summary,
            claims_entries,
            performance_metrics={"total_time": 5.2, "pipeline": "fast"},
        )
        print("✓ Report generation succeeded!")
    except Exception as e:
        print(f"✗ Report generation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify report content
    print("\n" + "-" * 60)
    print("REPORT CONTENT VERIFICATION")
    print("-" * 60)
    
    if not md_content:
        print("✗ Markdown report is empty")
        return False
    
    print(f"✓ Markdown report generated ({len(md_content)} chars)")
    
    if "Session" not in md_content:
        print("✗ Session section missing from report")
        return False
    print("✓ Session section present")
    
    if "Ingestion" not in md_content:
        print("✗ Ingestion section missing from report")
        return False
    print("✓ Ingestion section present")
    
    if "manual_text" not in md_content.lower():
        print("✗ Manual text extraction method not in report")
        return False
    print("✓ Manual text method tracked in report")
    
    if not html_content:
        print("✗ HTML report is empty")
        return False
    print(f"✓ HTML report generated ({len(html_content)} chars)")
    
    if not audit_json:
        print("✗ Audit JSON is empty")
        return False
    print(f"✓ Audit JSON generated ({len(audit_json)} chars)")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\nReport generation works correctly on Streamlit Cloud")
    print("with OCR disabled and manual text input only.")
    return True

if __name__ == "__main__":
    success = test_report_with_manual_text_ingestion()
    sys.exit(0 if success else 1)
