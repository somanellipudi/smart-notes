"""Regression test for json module usage in report generation."""

import json
from datetime import datetime

from src.reporting.research_report import (
    SessionMetadata,
    IngestionReport,
    VerificationSummary,
    ClaimEntry,
    build_report,
    save_reports,
)


def test_report_generation_does_not_shadow_json(tmp_path):
    session_metadata = SessionMetadata(
        session_id="test_session",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        seed=123,
        language_model="gpt-4",
        embedding_model="text-embedding-ada-002",
        nli_model="cross-encoder/qnli",
        inputs_used=["text"],
    )
    ingestion_report = IngestionReport(
        total_pages=0,
        pages_ocr=0,
        headers_removed=0,
        footers_removed=0,
        watermarks_removed=0,
        total_chunks=0,
        avg_chunk_size=0,
        extraction_methods=[],
    )
    verification_summary = VerificationSummary(
        total_claims=0,
        verified_count=0,
        low_confidence_count=0,
        rejected_count=0,
        avg_confidence=0.0,
        top_rejection_reasons=[],
    )

    md_content, html_content, audit_json = build_report(
        session_metadata,
        ingestion_report,
        verification_summary,
        claims=[],
        performance_metrics=None,
    )

    assert isinstance(md_content, str)
    assert isinstance(html_content, str)
    assert isinstance(audit_json, dict)
    assert callable(json.dumps)

    md_path, html_path, audit_path = save_reports(
        tmp_path,
        md_content,
        html_content,
        audit_json,
        prefix="report",
    )

    assert md_path.exists()
    assert audit_path.exists()
