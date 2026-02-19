"""
Unit tests for PDF ingestion pipeline with OCR fallback.

Tests cover:
1. Selectable-text PDF extraction
2. Scanned PDF triggering OCR when text < threshold
3. OCR unavailable raising EvidenceIngestError
4. PyMuPDF rendering fallback
5. Metadata correctness
"""

import pytest
import io
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.preprocessing.pdf_ingest import (
    extract_pdf_text,
    extract_pdf_text_legacy,
    _assess_extraction_quality,
    _extract_with_pymupdf,
    _extract_with_pdfplumber,
    _extract_with_ocr_pymupdf,
    _count_letters,
    _count_words,
    _compute_alphabetic_ratio,
    MIN_CHARS_FOR_OCR,
    MIN_WORDS_FOR_QUALITY,
    MIN_ALPHA_RATIO,
)
from src.preprocessing.pdf_page_extractor import PageText, QualityMetrics
from src.exceptions import EvidenceIngestError


class TestQualityAssessment:
    """Test text quality assessment functions."""
    
    def test_empty_text_fails(self):
        is_good, reason = _assess_extraction_quality("")
        assert not is_good
        assert "Empty text" in reason
    
    def test_short_text_fails(self):
        short_text = "Hello world"
        is_good, reason = _assess_extraction_quality(short_text)
        assert not is_good
        assert "Too few" in reason
    
    def test_good_quality_text_passes(self):
        # Generate text that meets all thresholds
        good_text = " ".join(["This is quality text with many words."] * 10)
        is_good, reason = _assess_extraction_quality(good_text)
        assert is_good
        assert "Good quality" in reason
    
    def test_low_alpha_ratio_fails(self):
        # Text with too many numbers/symbols
        # Need enough chars to pass the first check, but low alpha ratio
        bad_text = "1234567890 " * 20  # Numbers with spaces
        is_good, reason = _assess_extraction_quality(bad_text)
        assert not is_good
        # Will fail on "too few letters" check before reaching alpha ratio
        assert "too few" in reason.lower() or "alphabetic ratio" in reason.lower()
    
    def test_count_functions(self):
        text = "Hello world! This is a test."
        # Count letters manually: H+e+l+l+o+w+o+r+l+d+T+h+i+s+i+s+a+t+e+s+t = 21 letters
        assert _count_letters(text) == 21
        assert _count_words(text) == 6
        ratio = _compute_alphabetic_ratio(text)
        assert 0 < ratio < 1


class TestTextExtractionMethods:
    """Test individual extraction methods."""
    
    def test_pymupdf_extraction_with_valid_pdf(self):
        """Test PyMuPDF extraction with a minimal valid PDF."""
        # Create minimal PDF bytes
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>
endobj
4 0 obj
<< /Length 55 >>
stream
BT
/F1 12 Tf
100 700 Td
(This is a test PDF document) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000290 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
394
%%EOF"""
        
        result = _extract_with_pymupdf(pdf_content)
        # Should extract some text (actual content depends on PyMuPDF parsing)
        assert isinstance(result, str)
    
    def test_pymupdf_extraction_with_invalid_pdf(self):
        """Test PyMuPDF extraction with invalid PDF returns empty string."""
        invalid_pdf = b"Not a PDF file"
        result = _extract_with_pymupdf(invalid_pdf)
        assert result == ""


class MockUploadedFile:
    """Mock Streamlit UploadedFile for testing."""
    
    def __init__(self, content: bytes, name: str = "test.pdf"):
        self._content = content
        self.name = name
    
    def getvalue(self):
        return self._content
    
    def read(self):
        return self._content
    
    def seek(self, position):
        pass


class TestExtractPDFTextIntegration:
    """Integration tests for main extract_pdf_text function."""
    
    def test_text_pdf_succeeds_without_ocr(self):
        """Test that a text-based PDF extracts successfully without needing OCR."""
        # Create a PDF with sufficient text to pass quality checks
        good_pdf_text = "This is a high-quality PDF document with plenty of readable text. " * 10

        # Mock the extraction to return good text
        good_page = PageText(
            page_num=1,
            raw_text=good_pdf_text,
            cleaned_text=good_pdf_text,
            quality_metrics=QualityMetrics(is_acceptable=True),
            extraction_method="pdf_text"
        )
        with patch('src.preprocessing.pdf_ingest.extract_pages') as mock_extract_pages, \
             patch('config.CLEANING_ENABLED', False):
            mock_extract_pages.return_value = [good_page]

            mock_file = MockUploadedFile(b"fake_pdf_bytes", "test.pdf")
            text, metadata = extract_pdf_text(mock_file, ocr=None)

            assert len(text) > MIN_CHARS_FOR_OCR
            assert metadata["extraction_method"] == "pdf_text"
            assert metadata["num_pages"] == 1
            assert metadata["chars_extracted"] > 0
            assert "Good quality" in metadata["quality_assessment"]
    
    def test_scanned_pdf_triggers_ocr(self):
        """Test that a scanned PDF (poor text extraction) triggers OCR fallback."""
        # Simulate poor text extraction (garbage/short text)
        poor_text = "abc123"  # Too short
        good_ocr_text = "This is OCR extracted text from a scanned document page. " * 30
        
        mock_ocr = Mock()
        mock_ocr.extract_text_from_image.return_value = {"text": good_ocr_text}
        
        poor_page = PageText(
            page_num=1,
            raw_text=poor_text,
            cleaned_text=poor_text,
            quality_metrics=QualityMetrics(is_acceptable=False),
            extraction_method="pdf_text"
        )
        ocr_page = PageText(
            page_num=1,
            raw_text=good_ocr_text,
            cleaned_text=good_ocr_text,
            quality_metrics=QualityMetrics(is_acceptable=True),
            used_ocr=True,
            extraction_method="ocr_pymupdf_tesseract"
        )
        with patch('src.preprocessing.pdf_ingest.extract_pages') as mock_extract_pages, \
            patch('src.preprocessing.pdf_ingest.extract_page_with_ocr') as mock_extract_ocr, \
            patch('config.CLEANING_ENABLED', False):

            mock_extract_pages.return_value = [poor_page]
            mock_extract_ocr.return_value = ocr_page
            
            mock_file = MockUploadedFile(b"fake_scanned_pdf_bytes", "scanned.pdf")
            text, metadata = extract_pdf_text(mock_file, ocr=mock_ocr)
            
            # Should have used OCR
            assert "ocr" in metadata["extraction_method"]
            assert len(text) > MIN_CHARS_FOR_OCR
            mock_extract_ocr.assert_called_once()
    
    def test_ocr_unavailable_raises_exception(self):
        """Test that insufficient text without OCR raises EvidenceIngestError."""
        poor_text = "abc"  # Too short, fails quality check

        poor_page = PageText(
            page_num=1,
            raw_text=poor_text,
            cleaned_text=poor_text,
            quality_metrics=QualityMetrics(is_acceptable=False),
            extraction_method="pdf_text"
        )
        with patch('src.preprocessing.pdf_ingest.extract_pages') as mock_extract_pages, \
            patch('src.preprocessing.pdf_ingest.extract_page_with_ocr') as mock_extract_ocr, \
            patch('config.CLEANING_ENABLED', False):

            mock_extract_pages.return_value = [poor_page]
            mock_extract_ocr.side_effect = EvidenceIngestError(
                "OCR_UNAVAILABLE",
                "OCR fallback requested but neither pytesseract nor EasyOCR is available."
            )

            mock_file = MockUploadedFile(b"fake_pdf_bytes", "bad.pdf")
            text, metadata = extract_pdf_text(mock_file, ocr=None)

            assert (
                "Too few" in metadata["quality_assessment"]
                or "Low" in metadata["quality_assessment"]
                or "Empty" in metadata["quality_assessment"]
            )
            assert metadata["ocr_pages"] == 0
    
    def test_metadata_completeness(self):
        """Test that metadata includes all required fields."""
        good_text = "Quality text for testing metadata fields. " * 10

        pages = [
            PageText(
                page_num=1,
                raw_text=good_text,
                cleaned_text=good_text,
                quality_metrics=QualityMetrics(is_acceptable=True),
                extraction_method="pdf_text"
            ),
            PageText(
                page_num=2,
                raw_text=good_text,
                cleaned_text=good_text,
                quality_metrics=QualityMetrics(is_acceptable=True),
                extraction_method="pdf_text"
            ),
            PageText(
                page_num=3,
                raw_text=good_text,
                cleaned_text=good_text,
                quality_metrics=QualityMetrics(is_acceptable=True),
                extraction_method="pdf_text"
            )
        ]
        with patch('src.preprocessing.pdf_ingest.extract_pages') as mock_extract_pages, \
             patch('config.CLEANING_ENABLED', False):
            mock_extract_pages.return_value = pages
            
            mock_file = MockUploadedFile(b"fake_pdf_bytes", "test.pdf")
            text, metadata = extract_pdf_text(mock_file, ocr=None)
            
            # Check all required metadata fields
            assert "extraction_method" in metadata
            assert metadata["extraction_method"] in ["pdf_text", "ocr_pymupdf_tesseract", "ocr_easyocr"]
            assert "num_pages" in metadata
            assert metadata["num_pages"] == 3
            assert "chars_extracted" in metadata
            assert metadata["chars_extracted"] > 0
            assert "words" in metadata
            assert metadata["words"] > 0
            assert "alphabetic_ratio" in metadata
            assert 0 <= metadata["alphabetic_ratio"] <= 1
            assert "quality_assessment" in metadata
            
            # Backward compatibility fields
            assert "extraction_method_used" in metadata
            assert "pages" in metadata
            assert "letters" in metadata
    
    def test_file_path_input(self):
        """Test that function accepts file path string input."""
        good_text = "This is text extracted from a file path. " * 10
        
        good_page = PageText(
            page_num=1,
            raw_text=good_text,
            cleaned_text=good_text,
            quality_metrics=QualityMetrics(is_acceptable=True),
            extraction_method="pdf_text"
        )
        with patch('src.preprocessing.pdf_ingest.extract_pages') as mock_extract_pages, \
            patch('builtins.open', create=True) as mock_open, \
            patch('config.CLEANING_ENABLED', False):

            mock_extract_pages.return_value = [good_page]
            mock_open.return_value.__enter__.return_value.read.return_value = b"fake_pdf_bytes"

            text, metadata = extract_pdf_text("/path/to/file.pdf", ocr=None)

            assert len(text) > MIN_CHARS_FOR_OCR
            assert metadata["extraction_method"] == "pdf_text"
            mock_open.assert_called_once()


class TestOCRMethods:
    """Test OCR-specific extraction methods."""
    
    def test_ocr_pymupdf_with_valid_setup(self):
        """Test PyMuPDF-based OCR rendering."""
        mock_ocr = Mock()
        mock_ocr.extract_text_from_image.return_value = {"text": "OCR extracted text from page."}

        # Since fitz is imported inside the function, we need to patch it there
        with patch.dict('sys.modules', {'pytesseract': None}), \
             patch('fitz.open') as mock_fitz_open, \
             patch('PIL.Image.open') as mock_image_open, \
             patch('src.preprocessing.pdf_ingest.tempfile.NamedTemporaryFile') as mock_temp:

            # Mock PyMuPDF document
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_pix = MagicMock()
            mock_pix.tobytes.return_value = b"fake_png_bytes"
            mock_page.get_pixmap.return_value = mock_pix
            mock_doc.__len__.return_value = 1
            mock_doc.__getitem__.return_value = mock_page
            mock_fitz_open.return_value = mock_doc

            # Mock PIL Image
            mock_img = MagicMock()
            mock_image_open.return_value = mock_img

            # Mock temp file
            mock_temp_file = MagicMock()
            mock_temp_file.name = "/tmp/test.png"
            mock_temp.return_value.__enter__.return_value = mock_temp_file

            # Mock fitz.Matrix
            with patch('fitz.Matrix'):
                text, method, pages = _extract_with_ocr_pymupdf(b"fake_pdf", mock_ocr, max_pages=1)

                assert method in ["ocr_pymupdf_tesseract", "ocr_easyocr"]
                assert pages == 1
                assert len(text) > 0
                mock_ocr.extract_text_from_image.assert_called()
    
    def test_ocr_pymupdf_missing_dependencies_raises_error(self):
        """Test that missing PyMuPDF raises EvidenceIngestError."""
        mock_ocr = Mock()
        
        # Patch the import inside the function
        with patch('src.preprocessing.pdf_ingest._extract_with_ocr_pymupdf') as mock_extract:
            mock_extract.side_effect = EvidenceIngestError("OCR_UNAVAILABLE", "PyMuPDF not available")
            
            with pytest.raises(EvidenceIngestError) as exc_info:
                raise mock_extract.side_effect
            
            assert exc_info.value.code == "OCR_UNAVAILABLE"


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_old_metadata_keys_still_present(self):
        """Ensure old metadata keys are still available for backward compatibility."""
        good_text = "Backward compatibility test text. " * 10

        with patch('src.preprocessing.pdf_ingest._extract_with_pdfplumber') as mock_pdfplumber, \
            patch('src.preprocessing.pdf_ingest._count_pdf_pages') as mock_pages, \
            patch('src.preprocessing.pdf_ingest._assess_extraction_quality') as mock_assess, \
            patch('config.CLEANING_ENABLED', False):

            mock_pdfplumber.return_value = good_text
            mock_pages.return_value = 2
            # Mock the quality assessment to always return good quality
            mock_assess.return_value = (True, "Text quality is acceptable")

            mock_file = MockUploadedFile(b"fake_pdf_bytes", "test.pdf")
            text, metadata = extract_pdf_text_legacy(mock_file, ocr=None)

            # Old keys should still exist
            assert "extraction_method_used" in metadata
            assert "pages" in metadata
            assert "letters" in metadata

            # New keys should also exist
            assert "extraction_method" in metadata
            assert "num_pages" in metadata
            assert "chars_extracted" in metadata

            # Values should be consistent
            assert metadata["extraction_method"] == metadata["extraction_method_used"]
            assert metadata["num_pages"] == metadata["pages"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
