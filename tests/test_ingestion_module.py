"""
Unit tests for document ingestion and quality assessment.

Tests cover:
- Scanned PDF detection heuristics
- PDF extraction strategies (PyMuPDF, pdfplumber)  
- OCR fallback handling
- Image extraction
- Ingestion diagnostics
"""

import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

logger = logging.getLogger(__name__)

try:
    from src.ingestion.document_ingestor import (
        detect_scanned_or_low_text,
        IngestionDiagnostics,
        ingest_document,
        SCANNED_CHARS_THRESHOLD,
        SCANNED_SPACE_RATIO_THRESHOLD,
    )
    INGESTION_AVAILABLE = True
except ImportError:
    INGESTION_AVAILABLE = False


@pytest.mark.skipif(not INGESTION_AVAILABLE, reason="Ingestion module not available")
class TestScannedDetection:
    """Test scanned PDF detection heuristics."""
    
    def test_detect_scanned_empty_text(self):
        """Empty text should be detected as scanned."""
        is_scanned, metrics = detect_scanned_or_low_text("")
        assert is_scanned is True
        assert metrics["reason"] == "empty_text"
    
    def test_detect_scanned_short_text(self):
        """Text below threshold should be detected as scanned."""
        short_text = "Hello world. " * 5  # ~65 chars, below threshold
        is_scanned, metrics = detect_scanned_or_low_text(short_text)
        assert is_scanned is True
        assert metrics["reason"] == "text_too_short"
        assert metrics["length"] < SCANNED_CHARS_THRESHOLD
    
    def test_detect_good_text(self):
        """Normal text with good structure should not be scanned."""
        good_text = "This is a normal PDF with proper text extraction. " * 20
        is_scanned, metrics = detect_scanned_or_low_text(good_text)
        assert is_scanned is False
        assert metrics["reason"] == "looks_good"
    
    def test_detect_high_nonprintable_ratio(self):
        """Text with many nonprintable characters should be flagged."""
        # Create text with many non-printable characters
        bad_text = "Hello" + "\x00" * 50 + "World" * 10
        is_scanned, metrics = detect_scanned_or_low_text(bad_text)
        assert is_scanned is True
        assert metrics["reason"] == "high_nonprintable_ratio"
    
    def test_detect_low_space_ratio(self):
        """Text with very few spaces (unusual structure) should be flagged."""
        no_space_text = "abcdefghijklmnopqrstuvwxyz" * 100  # No spaces
        is_scanned, metrics = detect_scanned_or_low_text(no_space_text)
        assert is_scanned is True
        assert metrics["reason"] == "low_space_ratio"
        assert metrics["ratio"] < SCANNED_SPACE_RATIO_THRESHOLD


class TestIngestionDiagnostics:
    """Test ingestion diagnostics data structure."""
    
    def test_diagnostics_to_dict(self):
        """Diagnostics should convert to dictionary."""
        diag = IngestionDiagnostics(
            extracted_text_length=1000,
            scanned_detected=False,
            ocr_used=False,
            ocr_error=None,
            extraction_method="pymupdf",
            pages_processed=1,
            quality_score=0.8,
            first_300_chars="Sample text",
            warnings=[],
            errors=[]
        )
        
        diag_dict = diag.to_dict()
        assert diag_dict["extracted_text_length"] == 1000
        assert diag_dict["extraction_method"] == "pymupdf"
        assert diag_dict["quality_score"] == 0.8


class TestEvidenceValidation:
    """Test evidence store validation logic."""
    
    def test_ingestion_failure_message(self):
        """Should show useful error for ingestion failures."""
        from src.retrieval.evidence_store import EvidenceStore, get_ingestion_diagnostics
        
        store = EvidenceStore("test_session")
        # Empty store should fail validation
        
        is_valid, error_msg, classification = store.validate(min_chars=500)
        
        assert is_valid is False
        assert "Ingestion" in error_msg
        assert "No text" in error_msg
        assert classification == "NO_EVIDENCE"
    
    def test_ingestion_diagnostics_actionable(self):
        """Diagnostics should provide actionable suggestions."""
        from src.retrieval.evidence_store import EvidenceStore, get_ingestion_diagnostics
        
        store = EvidenceStore("test_session")
        diag = get_ingestion_diagnostics(store, min_chars=500)
        
        assert diag["is_valid"] is False
        assert "suggestions" in diag
        assert len(diag["suggestions"]) > 0
        # Should suggest OCR or other options
        suggestions_text = " ".join(diag["suggestions"]).lower()
        assert any(word in suggestions_text for word in ["ocr", "upload", "text"])


class TestPDFExtraction:
    """Test PDF extraction strategies (mocked)."""
    
    @patch("src.ingestion.document_ingestor.PYMUPDF_AVAILABLE", True)
    def test_pymupdf_extraction_success(self):
        """PyMuPDF should extract text from valid PDF."""
        from src.ingestion.document_ingestor import extract_text_from_pdf_pymupdf
        
        # Create mock PDF bytes with some content
        mock_pdf_bytes = b"mock_pdf_content"
        
        # For this test, we'll just verify the function signature works
        # (Full integration test would require real PDF or full mock)
        with patch("fitz.open") as mock_fitz:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = "Extracted text from page 1"
            mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
            mock_doc.close = MagicMock()
            mock_fitz.return_value = mock_doc
            
            text, meta, exc = extract_text_from_pdf_pymupdf(mock_pdf_bytes)
            
            assert text  # Should have extracted something
            assert exc is None
            assert meta.get("method") == "pymupdf"


class TestImageExtraction:
    """Test image extraction with OCR (mocked)."""
    
    def test_image_extraction_no_ocr(self):
        """Image extraction should fail gracefully without OCR."""
        from src.ingestion.document_ingestor import extract_text_from_image
        
        mock_image_bytes = b"mock_image_data"
        
        text, meta, exc = extract_text_from_image(mock_image_bytes, ocr_instance=None)
        
        assert text == ""
        assert exc is not None
        assert isinstance(exc, RuntimeError)
        assert "OCR" in str(exc)
    
    def test_image_extraction_with_ocr(self):
        """Image extraction should work with OCR instance."""
        from src.ingestion.document_ingestor import extract_text_from_image
        
        mock_image_bytes = b"mock_image_data"
        mock_ocr = MagicMock()
        mock_ocr.perform_ocr_bytes.return_value = "Extracted text from image"
        
        text, meta, exc = extract_text_from_image(mock_image_bytes, ocr_instance=mock_ocr)
        
        assert text == "Extracted text from image"
        assert exc is None
        assert meta.get("method") == "image_ocr"


class TestURLIngestion:
    """Test URL source ingestion."""
    
    def test_youtube_url_detection(self):
        """Should detect YouTube URLs."""
        from src.retrieval.url_ingest import detect_source_type
        
        urls = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
            ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
            ("https://example.com/article", "article"),
            ("https://arxiv.org/pdf/paper.pdf", "article"),
        ]
        
        for url, expected_type in urls:
            detected_type = detect_source_type(url)
            assert detected_type == expected_type, f"URL {url} should be {expected_type}, got {detected_type}"
    
    def test_youtube_extraction_unavailable(self):
        """Should handle missing youtube-transcript-api gracefully."""
        from src.retrieval.url_ingest import fetch_youtube_transcript
        
        with patch("src.retrieval.url_ingest.YOUTUBE_TRANSCRIPT_AVAILABLE", False):
            result = fetch_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            
            assert result.error is not None
            assert "not installed" in result.error.lower()
            assert result.text == ""
    
    def test_article_extraction_unavailable(self):
        """Should handle missing requests library gracefully."""
        from src.retrieval.url_ingest import fetch_article_content
        
        with patch("src.retrieval.url_ingest.REQUESTS_AVAILABLE", False):
            result = fetch_article_content("https://example.com/article")
            
            assert result.error is not None
            assert "not installed" in result.error.lower()
            assert result.text == ""


class TestIngestionIntegration:
    """Integration tests for full ingestion pipeline."""
    
    def test_ingest_document_no_ocr_for_pdf(self):
        """PDF ingestion should handle missing OCR."""
        mock_pdf_bytes = b"mock_pdf_content"
        
        with patch("src.ingestion.document_ingestor.PYMUPDF_AVAILABLE", False):
            with patch("src.ingestion.document_ingestor.PDFPLUMBER_AVAILABLE", False):
                text, diag = ingest_document(mock_pdf_bytes, "test.pdf", ocr_instance=None)
                
                assert text == ""
                assert diag.extracted_text_length == 0
                assert len(diag.errors) > 0
    
    def test_ingest_document_file_type_detection(self):
        """Should auto-detect file type from extension."""
        # This is a basic test, full integration would need real or mock file data
        mock_file_bytes = b"mock_data"
        
        with patch("src.ingestion.document_ingestor.PYMUPDF_AVAILABLE", False):
            with patch("src.ingestion.document_ingestor.PIL_AVAILABLE", False):
                # Test auto-detection works
                text, diag = ingest_document(mock_file_bytes, "test.pdf", file_type="auto")
                
                # Should detect as PDF and report failure appropriately
                assert diag.extraction_method in ["error", "pymupdf", "pdfplumber", "ocr"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
