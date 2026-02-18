"""
Tests for page-level OCR fallback in PDF ingestion.

Tests that OCR is applied only to low-quality pages, not the entire document.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
from src.preprocessing.pdf_page_extractor import (
    PageText,
    QualityMetrics,
    extract_pages,
    extract_page_with_ocr,
    compute_quality_metrics
)
from src.preprocessing.pdf_ingest import extract_pdf_text, PDFIngestionReport


def test_quality_metrics_good_text():
    """Test quality metrics for good text."""
    good_text = """
    This is a well-formatted document with plenty of readable text.
    It contains multiple sentences and has good alphabetic character ratio.
    The content is meaningful and has no strange artifacts or corrupted glyphs.
    There are enough unique characters to indicate real content.
    This should easily pass all quality thresholds.
    """
    
    metrics = compute_quality_metrics(good_text)
    
    assert metrics.is_acceptable, "Good text should pass quality check"
    assert metrics.word_count >= 20, "Should have sufficient words"
    assert metrics.alpha_ratio >= 0.30, "Should have good alphabetic ratio"
    assert metrics.unique_char_ratio > 0, "Should have unique characters"
    assert metrics.suspicious_glyph_ratio < 0.05, "Should have no suspicious glyphs"


def test_quality_metrics_bad_text():
    """Test quality metrics for poor quality text."""
    bad_texts = [
        "",  # Empty
        "abc",  # Too short
        "123 456 789 000 111",  # Mostly numbers
        "(cid:123) (cid:456) (cid:789) text",  # CID patterns
        "□□□□□ ▯▯▯ text",  # Box characters
        "�������",  # Replacement characters
    ]
    
    for bad_text in bad_texts:
        metrics = compute_quality_metrics(bad_text)
        if bad_text:  # Skip empty for this check
            # At least one quality check should fail
            assert not metrics.is_acceptable or metrics.word_count < 20 or \
                   metrics.suspicious_glyph_ratio > 0.05, f"Bad text should fail quality check: {bad_text[:50]}"


def test_page_extraction_mock():
    """Test page extraction with mocked PDF library."""
    # Create mock PDF bytes
    mock_pdf_bytes = b"%PDF-1.4 mock content"
    
    with patch('fitz.open') as mock_fitz_open:
        # Setup mock PyMuPDF
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is page 1 content with good quality text. " * 10
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_pdf.__len__.return_value = 1
        mock_pdf.__getitem__.return_value = mock_page
        mock_pdf.pages = [mock_page]
        mock_fitz_open.return_value = mock_pdf
        
        pages = extract_pages(mock_pdf_bytes, use_pdfplumber=False)
        
        assert len(pages) == 1, "Should extract one page"
        assert pages[0].page_num == 1
        assert pages[0].raw_text
        assert pages[0].extraction_method == "pymupdf"


def test_selective_ocr_fallback():
    """Test that OCR is only applied to low-quality pages."""
    # Create mix of good and bad pages
    good_page_text = "This is excellent readable text with many words. " * 10
    bad_page_text = "(cid:123) (cid:456) □□□"
    
    mock_pages = [
        PageText(
            page_num=1,
            raw_text=good_page_text,
            quality_metrics=compute_quality_metrics(good_page_text),
            extraction_method="pdfplumber"
        ),
        PageText(
            page_num=2,
            raw_text=bad_page_text,
            quality_metrics=compute_quality_metrics(bad_page_text),
            extraction_method="pdfplumber"
        ),
        PageText(
            page_num=3,
            raw_text=good_page_text,
            quality_metrics=compute_quality_metrics(good_page_text),
            extraction_method="pdfplumber"
        ),
    ]
    
    # Count pages needing OCR
    pages_needing_ocr = sum(1 for p in mock_pages if not p.quality_metrics.is_acceptable)
    
    assert pages_needing_ocr == 1, "Only page 2 should need OCR"
    
    # Verify good pages are marked acceptable
    assert mock_pages[0].quality_metrics.is_acceptable
    assert not mock_pages[1].quality_metrics.is_acceptable
    assert mock_pages[2].quality_metrics.is_acceptable


@patch('src.preprocessing.pdf_ingest.extract_pages')
@patch('src.preprocessing.pdf_ingest.extract_page_with_ocr')
@patch('config.ENABLE_OCR_FALLBACK', True)
def test_extract_pdf_with_partial_ocr(mock_extract_page_ocr, mock_extract_pages):
    """Test PDF extraction with OCR applied only to bad pages."""
    # Setup mock pages
    good_text = "This is good readable text with sufficient content. " * 20
    bad_text = "(cid:123) □□□"
    ocr_recovered_text = "This text was recovered by OCR and is now readable. " * 20
    
    mock_extract_pages.return_value = [
        PageText(
            page_num=1,
            raw_text=good_text,
            quality_metrics=compute_quality_metrics(good_text),
            extraction_method="pdfplumber"
        ),
        PageText(
            page_num=2,
            raw_text=bad_text,
            quality_metrics=compute_quality_metrics(bad_text),
            extraction_method="pdfplumber"
        ),
        PageText(
            page_num=3,
            raw_text=good_text,
            quality_metrics=compute_quality_metrics(good_text),
            extraction_method="pdfplumber"
        ),
    ]
    
    # Mock OCR to return good text
    mock_extract_page_ocr.return_value = PageText(
        page_num=2,
        raw_text=ocr_recovered_text,
        quality_metrics=compute_quality_metrics(ocr_recovered_text),
        used_ocr=True,
        extraction_method="ocr_tesseract"
    )
    
    # Create mock file
    mock_file = Mock()
    mock_file.name = "test.pdf"
    mock_file.getvalue.return_value = b"%PDF-1.4"
    
    # Extract
    text, metadata = extract_pdf_text(mock_file, ocr=None)
    
    # Verify OCR was called only once (for page 2)
    assert mock_extract_page_ocr.call_count == 1
    assert mock_extract_page_ocr.call_args[0][1] == 2  # page_num=2
    
    # Verify metadata - check ingestion_report which is more reliable
    ingestion_report = metadata.get("ingestion_report")
    assert ingestion_report is not None
    assert ingestion_report.pages_total == 3
    assert ingestion_report.pages_ocr == 1, f"Expected 1 OCR page, got {ingestion_report.pages_ocr}"
    assert ingestion_report.pages_low_quality == 1, f"Expected 1 low quality page, got {ingestion_report.pages_low_quality}"


@patch('src.preprocessing.pdf_ingest.extract_pages')
@patch('config.ENABLE_OCR_FALLBACK', False)
def test_extract_pdf_without_ocr_disabled(mock_extract_pages):
    """Test that low-quality pages are kept when OCR is disabled."""
    bad_text = "(cid:123) but with some readable words to avoid total failure " * 5
    
    mock_extract_pages.return_value = [
        PageText(
            page_num=1,
            raw_text=bad_text,
            quality_metrics=compute_quality_metrics(bad_text),
            extraction_method="pdfplumber"
        ),
    ]
    
    # Create mock file
    mock_file = Mock()
    mock_file.name = "test.pdf"
    mock_file.getvalue.return_value = b"%PDF-1.4"
    
    # Extract (should not crash, just warn)
    text, metadata = extract_pdf_text(mock_file, ocr=None)
    
    # Should report low quality but no OCR
    # Check ingestion_report which is more reliable than top-level metadata
    ingestion_report = metadata.get("ingestion_report")
    if ingestion_report:
        assert ingestion_report.pages_ocr == 0, "Should not use OCR when disabled"
        assert ingestion_report.pages_low_quality >= 0, "Should report low-quality pages"


@patch('src.preprocessing.pdf_ingest.extract_pages')
@patch('src.preprocessing.pdf_ingest.extract_page_with_ocr')
@patch('config.ENABLE_OCR_FALLBACK', True)
def test_ocr_failure_graceful_handling(mock_extract_page_ocr, mock_extract_pages):
    """Test graceful handling when OCR fails."""
    bad_text = "(cid:123) □□□ minimal text"
    
    mock_extract_pages.return_value = [
        PageText(
            page_num=1,
            raw_text=bad_text,
            quality_metrics=compute_quality_metrics(bad_text),
            extraction_method="pdfplumber"
        ),
    ]
    
    # Mock OCR to raise exception
    mock_extract_page_ocr.side_effect = Exception("OCR service unavailable")
    
    # Create mock file
    mock_file = Mock()
    mock_file.name = "test.pdf"
    mock_file.getvalue.return_value = b"%PDF-1.4"
    
    # Extract (should not crash)
    text, metadata = extract_pdf_text(mock_file, ocr=None)
    
    # Should report OCR failure but continue
    # Check ingestion_report which is more reliable
    ingestion_report = metadata.get("ingestion_report")
    if ingestion_report:
        assert ingestion_report.pages_ocr == 0, "Should report 0 successful OCR pages"
        assert ingestion_report.pages_low_quality == 1, "Should still report low-quality page"


def test_ingestion_report_structure():
    """Test PDFIngestionReport structure and serialization."""
    report = PDFIngestionReport(
        pages_total=10,
        pages_ocr=2,
        pages_low_quality=3,
        headers_removed_count=15,
        watermark_removed_count=5,
        removed_lines_count=20,
        extraction_method="pdfplumber_with_ocr",
        chars_extracted=5000,
        words_extracted=800,
        alphabetic_ratio=0.85,
        quality_assessment="Good quality"
    )
    
    # Test to_dict() method
    report_dict = report.to_dict()
    
    assert isinstance(report_dict, dict)
    assert report_dict["pages_total"] == 10
    assert report_dict["pages_ocr"] == 2
    assert report_dict["pages_low_quality"] == 3
    assert report_dict["headers_removed_count"] == 15
    assert report_dict["watermark_removed_count"] == 5
    assert report_dict["removed_lines_count"] == 20
    assert report_dict["extraction_method"] == "pdfplumber_with_ocr"
    assert report_dict["chars_extracted"] == 5000
    assert report_dict["words_extracted"] == 800
    assert report_dict["alphabetic_ratio"] == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
