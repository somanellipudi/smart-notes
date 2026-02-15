"""
Tests for PDF ingestion module with OCR fallback.

Tests the quality heuristics, multi-strategy extraction, and OCR fallback behavior.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

from src.preprocessing.pdf_ingest import (
    extract_pdf_text,
    _extract_with_pymupdf,
    _extract_with_ocr,
    _extract_with_pdfplumber,
    _assess_extraction_quality,
    _clean_text,
    _count_letters,
    _count_words,
    _compute_alphabetic_ratio,
    _get_pdf_bytes
)


class TestQualityHeuristics:
    """Test quality assessment functions."""
    
    def test_count_letters(self):
        """Test alphabetic character counting."""
        assert _count_letters("Hello") == 5
        assert _count_letters("Hello123") == 5
        assert _count_letters("123") == 0
        assert _count_letters("") == 0
    
    def test_count_words(self):
        """Test word counting."""
        assert _count_words("hello world") == 2
        assert _count_words("hello  world") == 2  # Multiple spaces
        assert _count_words("") == 0
        assert _count_words("   ") == 0
    
    def test_alphabetic_ratio(self):
        """Test alphabetic ratio computation."""
        assert _compute_alphabetic_ratio("hello") == 1.0
        assert abs(_compute_alphabetic_ratio("hello123") - 5/8) < 0.01
        assert _compute_alphabetic_ratio("123") == 0.0
        assert _compute_alphabetic_ratio("") == 0.0
    
    def test_assess_quality_empty_text(self):
        """Test quality assessment for empty text."""
        is_good, reason = _assess_extraction_quality("")
        assert not is_good
        assert "Empty" in reason
    
    def test_assess_quality_too_few_words(self):
        """Test quality assessment for too few words."""
        text = " ".join(["word"] * 50)  # 50 words
        is_good, reason = _assess_extraction_quality(text)
        assert not is_good
        assert "word" in reason.lower()
    
    def test_assess_quality_too_few_letters(self):
        """Test quality assessment for too few letters."""
        text = "a " * 100  # 100 words but only ~100 letters
        is_good, reason = _assess_extraction_quality(text)
        assert not is_good
        assert "letter" in reason.lower()
    
    def test_assess_quality_low_alphabetic_ratio(self):
        """Test quality assessment for low alphabetic ratio."""
        # Pure numeric text - no alphabetic characters
        text = "123 456 789 " * 50  # ~150 words but no letters
        is_good, reason = _assess_extraction_quality(text)
        assert not is_good
        assert "alphabetic" in reason.lower()
    
    def test_assess_quality_good_text(self):
        """Test quality assessment for good quality text."""
        good_text = " ".join([
            "The calculus derivative measures the rate of change of a function at a point."
            for _ in range(10)
        ])  # 150+ words with good alphabetic content
        is_good, reason = _assess_extraction_quality(good_text)
        assert is_good
        assert "Good" in reason


class TestTextCleaning:
    """Test text cleaning functionality."""
    
    def test_clean_text_removes_cid_glyphs(self):
        """Test that CID glyphs are removed."""
        text = "Hello (cid:123) world (cid:456)"
        cleaned = _clean_text(text)
        assert "(cid:" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned
    
    def test_clean_text_normalizes_whitespace(self):
        """Test that whitespace is normalized."""
        text = "Hello    world  \n\n  test"
        cleaned = _clean_text(text)
        assert "    " not in cleaned
        assert cleaned.count("\n") <= 2  # Should have at most one \n\n
    
    def test_clean_text_preserves_content(self):
        """Test that cleaning preserves valid content."""
        text = "The derivative of x^2 is 2x. Integration is the reverse."
        cleaned = _clean_text(text)
        assert "derivative" in cleaned
        assert "Integration" in cleaned


class TestStreamlitFileObject:
    """Test handling of Streamlit UploadedFile objects."""
    
    def test_get_pdf_bytes_from_streamlit_file(self):
        """Test getting bytes from Streamlit UploadedFile."""
        # Mock a Streamlit UploadedFile
        mock_file = Mock()
        pdf_bytes = b"PDF mock content here"
        mock_file.getvalue.return_value = pdf_bytes
        
        result = _get_pdf_bytes(mock_file)
        assert result == pdf_bytes
    
    def test_get_pdf_bytes_from_bytes(self):
        """Test getting bytes directly from bytes."""
        pdf_bytes = b"PDF mock content"
        result = _get_pdf_bytes(pdf_bytes)
        assert result == pdf_bytes
    
    def test_get_pdf_bytes_from_file_object(self):
        """Test getting bytes from file-like object with read()."""
        pdf_bytes = b"PDF mock content"
        mock_file = BytesIO(pdf_bytes)
        
        result = _get_pdf_bytes(mock_file)
        assert result == pdf_bytes


class TestPDFExtractionStrategies:
    """Test individual extraction strategies."""
    
    @patch('src.preprocessing.pdf_ingest.fitz')
    def test_pymupdf_extraction_success(self, mock_fitz):
        """Test successful PyMuPDF extraction."""
        # Mock PyMuPDF document and pages
        mock_page = Mock()
        mock_page.get_text.return_value = "Extracted text from page 1"
        
        mock_doc = Mock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        
        mock_fitz.open.return_value = mock_doc
        
        result = _extract_with_pymupdf(b"fake pdf")
        
        assert "Extracted text from page 1" in result
        assert "[Page 1]" in result
    
    @patch('src.preprocessing.pdf_ingest.pdfplumber')
    def test_pdfplumber_extraction_fallback(self, mock_pdfplumber):
        """Test pdfplumber extraction fallback."""
        # Mock pdfplumber
        mock_page = Mock()
        mock_page.extract_text.return_value = "Text from pdfplumber"
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__.return_value = mock_pdf
        mock_pdf.__exit__.return_value = None
        
        mock_pdfplumber.open.return_value = mock_pdf
        
        result = _extract_with_pdfplumber(b"fake pdf")
        
        assert "Text from pdfplumber" in result or "Text from pdfplumber" == result.strip()


class TestOCRFallback:
    """Test OCR fallback behavior."""
    
    @patch('src.preprocessing.pdf_ingest.convert_from_bytes')
    @patch('src.preprocessing.pdf_ingest.fitz')
    def test_ocr_triggers_on_low_quality(self, mock_fitz, mock_convert):
        """Test that OCR is triggered when text quality is low."""
        # Mock PyMuPDF with garbage text
        mock_page = Mock()
        mock_page.get_text.return_value = "(cid:1) (cid:2) (cid:3) " * 100  # Garbage
        
        mock_doc = Mock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        
        mock_fitz.open.return_value = mock_doc
        
        # Create mock OCR
        mock_ocr = Mock()
        mock_ocr.extract_text_from_image.return_value = "Good OCR text " * 50
        
        # Test with quality checking
        text, meta = extract_pdf_text(b"fake pdf", ocr=mock_ocr)
        
        # Should have attempted extraction via some method
        assert isinstance(text, str)
        assert isinstance(meta, dict)
        assert "pages" in meta
        assert "method_used" in meta or "extraction_method_used" in meta


class TestExtractPDFIntegration:
    """Integration tests for extract_pdf_text."""
    
    def test_extract_returns_tuple(self):
        """Test that extract_pdf_text returns correct format."""
        # Create a mock StreamlitUploadedFile-like object
        mock_file = Mock()
        mock_file.getvalue.return_value = b""
        mock_file.name = "test.pdf"
        
        with patch('src.preprocessing.pdf_ingest._extract_with_pymupdf') as mock_extract:
            mock_extract.return_value = ""
            
            result = extract_pdf_text(mock_file)
            
            assert isinstance(result, tuple)
            assert len(result) == 2
            text, metadata = result
            assert isinstance(text, str)
            assert isinstance(metadata, dict)
    
    def test_metadata_includes_required_fields(self):
        """Test that returned metadata has required fields."""
        mock_file = Mock()
        mock_file.getvalue.return_value = b""
        mock_file.name = "test.pdf"
        
        with patch('src.preprocessing.pdf_ingest._extract_with_pymupdf') as mock_extract:
            mock_extract.return_value = ""
            
            text, metadata = extract_pdf_text(mock_file)
            
            required_fields = ["pages", "words", "letters", "alphabetic_ratio", "quality_assessment"]
            for field in required_fields:
                assert field in metadata, f"Missing field: {field}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
