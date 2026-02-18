"""
JSON audit schema validation tests.

Validates that the audit JSON output conforms to the expected schema,
includes all required fields, and has correct data types.
"""

import pytest
import json
from datetime import datetime
from src.reporting.research_report import (
    ResearchReportBuilder,
    SessionMetadata,
    IngestionReport,
    VerificationSummary,
    ClaimEntry,
)


@pytest.fixture
def complete_report_builder():
    """Create a complete report builder with all sections populated."""
    builder = ResearchReportBuilder()
    
    builder.add_session_metadata(
        SessionMetadata(
            session_id="audit_test_001",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            seed=42,
            language_model="gpt-4",
            embedding_model="text-embedding-ada-002",
            nli_model="cross-encoder/qnli",
            inputs_used=["source1.pdf", "source2.txt"],
        )
    )
    
    builder.add_ingestion_report(
        IngestionReport(
            total_pages=50,
            pages_ocr=45,
            headers_removed=49,
            footers_removed=49,
            watermarks_removed=2,
            total_chunks=225,
            avg_chunk_size=512,
            extraction_methods=["pdf_text", "ocr"],
        )
    )
    
    builder.add_verification_summary(
        VerificationSummary(
            total_claims=30,
            verified_count=24,
            low_confidence_count=4,
            rejected_count=2,
            avg_confidence=0.85,
            top_rejection_reasons=[
                ("No evidence", 1),
                ("Contradicted", 1),
            ],
            calibration_metrics={
                "ece": 0.042,
                "brier": 0.125,
            },
        )
    )
    
    builder.add_claims([
        ClaimEntry(
            claim_text="Test claim 1",
            status="VERIFIED",
            confidence=0.95,
            evidence_count=3,
            top_evidence="Evidence for claim 1",
            page_num=10,
            span_id="p10_s1",
        ),
        ClaimEntry(
            claim_text="Test claim 2",
            status="LOW_CONFIDENCE",
            confidence=0.60,
            evidence_count=1,
            top_evidence="Weak evidence",
            page_num=20,
        ),
        ClaimEntry(
            claim_text="Test claim 3",
            status="REJECTED",
            confidence=0.10,
            evidence_count=0,
            top_evidence="No evidence",
        ),
    ])
    
    builder.add_performance_metrics({
        "total_time_sec": 120.5,
        "inference_time_sec": 95.3,
        "memory_mb": 2048,
    })
    
    return builder


class TestAuditJSONSchema:
    """Test audit JSON schema compliance."""

    def test_audit_json_top_level_structure(self, complete_report_builder):
        """Test that audit JSON has required top-level structure."""
        audit = complete_report_builder.build_audit_json()
        
        required_fields = [
            "report_type",
            "generated_at",
            "session",
            "ingestion",
            "verification",
            "claims",
            "performance",
            "metadata",
        ]
        
        for field in required_fields:
            assert field in audit, f"Missing required field: {field}"

    def test_report_type_is_audit(self, complete_report_builder):
        """Test that report_type is 'audit'."""
        audit = complete_report_builder.build_audit_json()
        assert audit["report_type"] == "audit"

    def test_generated_at_is_iso_timestamp(self, complete_report_builder):
        """Test that generated_at is a valid ISO timestamp."""
        audit = complete_report_builder.build_audit_json()
        generated_at = audit["generated_at"]
        
        # Try to parse as ISO timestamp
        try:
            datetime.fromisoformat(generated_at)
        except (ValueError, TypeError):
            pytest.fail(f"generated_at is not valid ISO timestamp: {generated_at}")

    def test_metadata_structure(self, complete_report_builder):
        """Test metadata field structure."""
        audit = complete_report_builder.build_audit_json()
        metadata = audit["metadata"]
        
        assert "report_version" in metadata
        assert "total_sections" in metadata
        assert metadata["report_version"] == "1.0"
        assert metadata["total_sections"] == 6

    def test_session_structure(self, complete_report_builder):
        """Test session object structure."""
        audit = complete_report_builder.build_audit_json()
        session = audit["session"]
        
        required_session_fields = [
            "session_id",
            "timestamp",
            "version",
            "seed",
            "language_model",
            "embedding_model",
            "nli_model",
            "inputs_used",
        ]
        
        for field in required_session_fields:
            assert field in session, f"Missing session field: {field}"

    def test_session_types(self, complete_report_builder):
        """Test session field types."""
        audit = complete_report_builder.build_audit_json()
        session = audit["session"]
        
        assert isinstance(session["session_id"], str)
        assert isinstance(session["timestamp"], str)
        assert isinstance(session["version"], str)
        assert isinstance(session["seed"], int)
        assert isinstance(session["language_model"], str)
        assert isinstance(session["embedding_model"], str)
        assert isinstance(session["nli_model"], str)
        assert isinstance(session["inputs_used"], list)

    def test_ingestion_structure(self, complete_report_builder):
        """Test ingestion object structure."""
        audit = complete_report_builder.build_audit_json()
        ingestion = audit["ingestion"]
        
        required_ingestion_fields = [
            "total_pages",
            "pages_ocr",
            "headers_removed",
            "footers_removed",
            "watermarks_removed",
            "total_chunks",
            "avg_chunk_size",
            "extraction_methods",
        ]
        
        for field in required_ingestion_fields:
            assert field in ingestion, f"Missing ingestion field: {field}"

    def test_ingestion_types(self, complete_report_builder):
        """Test ingestion field types."""
        audit = complete_report_builder.build_audit_json()
        ingestion = audit["ingestion"]
        
        assert isinstance(ingestion["total_pages"], int)
        assert isinstance(ingestion["pages_ocr"], int)
        assert isinstance(ingestion["headers_removed"], int)
        assert isinstance(ingestion["footers_removed"], int)
        assert isinstance(ingestion["watermarks_removed"], int)
        assert isinstance(ingestion["total_chunks"], int)
        assert isinstance(ingestion["avg_chunk_size"], int)
        assert isinstance(ingestion["extraction_methods"], list)

    def test_ingestion_counts_valid(self, complete_report_builder):
        """Test that ingestion counts are valid (e.g., OCR <= total pages)."""
        audit = complete_report_builder.build_audit_json()
        ingestion = audit["ingestion"]
        
        assert ingestion["pages_ocr"] <= ingestion["total_pages"]
        assert ingestion["headers_removed"] <= ingestion["total_pages"]
        assert ingestion["footers_removed"] <= ingestion["total_pages"]

    def test_verification_structure(self, complete_report_builder):
        """Test verification object structure."""
        audit = complete_report_builder.build_audit_json()
        verification = audit["verification"]
        
        required_verification_fields = [
            "total_claims",
            "verified_count",
            "low_confidence_count",
            "rejected_count",
            "avg_confidence",
            "top_rejection_reasons",
            "calibration_metrics",
        ]
        
        for field in required_verification_fields:
            assert field in verification, f"Missing verification field: {field}"

    def test_verification_types(self, complete_report_builder):
        """Test verification field types."""
        audit = complete_report_builder.build_audit_json()
        verification = audit["verification"]
        
        assert isinstance(verification["total_claims"], int)
        assert isinstance(verification["verified_count"], int)
        assert isinstance(verification["low_confidence_count"], int)
        assert isinstance(verification["rejected_count"], int)
        assert isinstance(verification["avg_confidence"], float)
        assert isinstance(verification["top_rejection_reasons"], list)
        assert isinstance(verification["calibration_metrics"], (dict, type(None)))

    def test_verification_counts_consistency(self, complete_report_builder):
        """Test that verification counts are consistent."""
        audit = complete_report_builder.build_audit_json()
        verification = audit["verification"]
        
        total = (
            verification["verified_count"]
            + verification["low_confidence_count"]
            + verification["rejected_count"]
        )
        assert total == verification["total_claims"], (
            f"Claim counts don't sum: {verification['verified_count']} + "
            f"{verification['low_confidence_count']} + "
            f"{verification['rejected_count']} != {verification['total_claims']}"
        )

    def test_confidence_range(self, complete_report_builder):
        """Test that confidence scores are in valid range [0, 1]."""
        audit = complete_report_builder.build_audit_json()
        verification = audit["verification"]
        
        assert 0 <= verification["avg_confidence"] <= 1, (
            f"Average confidence {verification['avg_confidence']} out of range [0, 1]"
        )

    def test_rejection_reasons_structure(self, complete_report_builder):
        """Test rejection reasons structure."""
        audit = complete_report_builder.build_audit_json()
        verification = audit["verification"]
        
        if verification["top_rejection_reasons"]:
            for reason_tuple in verification["top_rejection_reasons"]:
                assert isinstance(reason_tuple, (list, tuple))
                assert len(reason_tuple) == 2
                assert isinstance(reason_tuple[0], str)
                assert isinstance(reason_tuple[1], int)

    def test_calibration_metrics_types(self, complete_report_builder):
        """Test calibration metrics field types."""
        audit = complete_report_builder.build_audit_json()
        verification = audit["verification"]
        
        if verification["calibration_metrics"]:
            for key, value in verification["calibration_metrics"].items():
                assert isinstance(key, str)
                assert isinstance(value, (int, float))

    def test_claims_array_structure(self, complete_report_builder):
        """Test claims array structure."""
        audit = complete_report_builder.build_audit_json()
        claims = audit["claims"]
        
        assert isinstance(claims, list)
        assert len(claims) > 0

    def test_claim_entry_structure(self, complete_report_builder):
        """Test individual claim entry structure."""
        audit = complete_report_builder.build_audit_json()
        claims = audit["claims"]
        
        required_claim_fields = [
            "claim_text",
            "status",
            "confidence",
            "evidence_count",
            "top_evidence",
            "page_num",
            "span_id",
        ]
        
        for claim in claims:
            for field in required_claim_fields:
                assert field in claim, f"Missing claim field: {field}"

    def test_claim_entry_types(self, complete_report_builder):
        """Test claim entry field types."""
        audit = complete_report_builder.build_audit_json()
        claims = audit["claims"]
        
        for claim in claims:
            assert isinstance(claim["claim_text"], str)
            assert isinstance(claim["status"], str)
            assert isinstance(claim["confidence"], float)
            assert isinstance(claim["evidence_count"], int)
            assert isinstance(claim["top_evidence"], str)
            assert isinstance(claim["page_num"], (int, type(None)))
            assert isinstance(claim["span_id"], (str, type(None)))

    def test_claim_confidence_range(self, complete_report_builder):
        """Test that claim confidence scores are in valid range."""
        audit = complete_report_builder.build_audit_json()
        claims = audit["claims"]
        
        for claim in claims:
            conf = claim["confidence"]
            assert 0 <= conf <= 1, (
                f"Claim confidence {conf} out of range [0, 1]: {claim['claim_text']}"
            )

    def test_claim_status_values(self, complete_report_builder):
        """Test that claim status has valid values."""
        audit = complete_report_builder.build_audit_json()
        claims = audit["claims"]
        
        valid_statuses = {"VERIFIED", "LOW_CONFIDENCE", "REJECTED"}
        
        for claim in claims:
            status = claim["status"]
            assert status in valid_statuses, (
                f"Invalid claim status: {status}. Must be one of {valid_statuses}"
            )

    def test_performance_structure(self, complete_report_builder):
        """Test performance metrics structure."""
        audit = complete_report_builder.build_audit_json()
        performance = audit["performance"]
        
        assert isinstance(performance, dict)

    def test_performance_types(self, complete_report_builder):
        """Test performance metrics field types."""
        audit = complete_report_builder.build_audit_json()
        performance = audit["performance"]
        
        for key, value in performance.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float, str))


class TestAuditJSONFormatCompliance:
    """Test JSON format compliance for serialization."""

    def test_audit_json_is_fully_serializable(self, complete_report_builder):
        """Test that audit JSON can be fully serialized."""
        audit = complete_report_builder.build_audit_json()
        
        # Should not raise
        json_str = json.dumps(audit)
        assert isinstance(json_str, str)

    def test_audit_json_round_trip(self, complete_report_builder):
        """Test that audit JSON can be serialized and deserialized."""
        audit = complete_report_builder.build_audit_json()
        
        json_str = json.dumps(audit)
        reloaded = json.loads(json_str)
        
        assert reloaded["report_type"] == audit["report_type"]
        assert len(reloaded["claims"]) == len(audit["claims"])

    def test_audit_json_size_reasonable(self, complete_report_builder):
        """Test that JSON size is reasonable."""
        audit = complete_report_builder.build_audit_json()
        json_str = json.dumps(audit)
        
        # Should be less than 1MB
        size_mb = len(json_str) / (1024 * 1024)
        assert size_mb < 1, f"JSON too large: {size_mb:.1f}MB"

    def test_audit_json_no_nan_or_inf(self, complete_report_builder):
        """Test that JSON doesn't contain NaN or Infinity."""
        audit = complete_report_builder.build_audit_json()
        json_str = json.dumps(audit)
        
        # JSON spec doesn't allow NaN or Infinity
        assert "NaN" not in json_str
        assert "Infinity" not in json_str
        assert "-Infinity" not in json_str


class TestAuditJSONEdgeCases:
    """Test edge cases in audit JSON generation."""

    def test_audit_json_minimal_session(self):
        """Test audit JSON with minimal session metadata."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(
            SessionMetadata(
                session_id="minimal",
                timestamp=datetime.now().isoformat(),
                version="1.0",
                seed=0,
                language_model="test",
                embedding_model="test",
                nli_model="test",
                inputs_used=[],
            )
        )
        
        audit = builder.build_audit_json()
        assert audit["session"]["session_id"] == "minimal"
        assert audit["session"]["inputs_used"] == []

    def test_audit_json_empty_claims(self):
        """Test audit JSON with no claims."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(
            SessionMetadata(
                session_id="no_claims",
                timestamp=datetime.now().isoformat(),
                version="1.0",
                seed=0,
                language_model="test",
                embedding_model="test",
                nli_model="test",
                inputs_used=["test.pdf"],
            )
        )
        builder.add_claims([])
        
        audit = builder.build_audit_json()
        assert audit["claims"] == []

    def test_audit_json_all_rejected_claims(self):
        """Test audit JSON when all claims are rejected."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(
            SessionMetadata(
                session_id="all_rejected",
                timestamp=datetime.now().isoformat(),
                version="1.0",
                seed=0,
                language_model="test",
                embedding_model="test",
                nli_model="test",
                inputs_used=["test.pdf"],
            )
        )
        builder.add_verification_summary(
            VerificationSummary(
                total_claims=3,
                verified_count=0,
                low_confidence_count=0,
                rejected_count=3,
                avg_confidence=0.1,
                top_rejection_reasons=[("No evidence", 3)],
            )
        )
        builder.add_claims([
            ClaimEntry(
                claim_text=f"Rejected claim {i}",
                status="REJECTED",
                confidence=0.05,
                evidence_count=0,
                top_evidence="No evidence",
            )
            for i in range(3)
        ])
        
        audit = builder.build_audit_json()
        verification = audit["verification"]
        
        assert verification["rejected_count"] == 3
        assert verification["verified_count"] == 0
        assert all(c["status"] == "REJECTED" for c in audit["claims"])

    def test_audit_json_special_characters_preserved(self):
        """Test that special characters in claims are preserved in JSON."""
        builder = ResearchReportBuilder()
        builder.add_session_metadata(
            SessionMetadata(
                session_id="special_chars",
                timestamp=datetime.now().isoformat(),
                version="1.0",
                seed=0,
                language_model="test",
                embedding_model="test",
                nli_model="test",
                inputs_used=["测试.pdf", "test_émojis_🎓.txt"],
            )
        )
        builder.add_verification_summary(
            VerificationSummary(
                total_claims=1,
                verified_count=1,
                low_confidence_count=0,
                rejected_count=0,
                avg_confidence=0.9,
                top_rejection_reasons=[],
            )
        )
        special_claim = ClaimEntry(
            claim_text='E=mc², "quotes" & special chars | ñ, é, 中文',
            status="VERIFIED",
            confidence=0.9,
            evidence_count=1,
            top_evidence='Evidence with "quotes" and unicode',
        )
        builder.add_claims([special_claim])
        
        audit = builder.build_audit_json()
        json_str = json.dumps(audit)  # Should not raise
        reloaded = json.loads(json_str)
        
        assert reloaded["claims"][0]["claim_text"] == special_claim.claim_text
        # Verify special characters preserved in inputs and claim text
        assert "测试" in reloaded["session"]["inputs_used"][0]
        assert "中文" in reloaded["claims"][0]["claim_text"]
        assert "🎓" in reloaded["session"]["inputs_used"][1]
