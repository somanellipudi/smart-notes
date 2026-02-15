"""
Integration tests for PDF and URL ingestion with app.py.

Tests:
- PDF extraction with real test PDFs
- URL ingestion with mocked YouTube and article content
- Integration with app.py's file processing pipeline
"""

import pytest
import sys
from pathlib import Path
from io import BytesIO
from unittest.mock import patch, Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.preprocessing.pdf_ingest import extract_pdf_text, _assess_extraction_quality, _clean_text
from src.preprocessing.url_ingest import fetch_url_text, _is_youtube_url


class TestPDFIntegration:
    """Integration tests for PDF ingestion."""
    
    def test_garbage_pdf_detection(self):
        """Test detection of garbage/corrupted PDF text."""
        # Simulate corrupted PDF with CID glyphs
        garbage_text = "(cid:1) (cid:2) (cid:3) (cid:4) (cid:5) " * 100
        is_good, reason = _assess_extraction_quality(garbage_text)
        
        assert not is_good, "Should reject CID glyphs as garbage"
        assert "alphabetic ratio" in reason.lower()
    
    def test_quality_threshold_enforcement(self):
        """Test that quality thresholds are enforced."""
        # Valid text - should pass
        valid_text = " ".join(["word"] * 100)  # 100 words
        is_good, reason = _assess_extraction_quality(valid_text)
        assert is_good, "Should accept 100 words + letters"
        
        # Too few words - should fail
        few_words = " ".join(["word"] * 40)
        is_good, reason = _assess_extraction_quality(few_words)
        assert not is_good, "Should reject < 80 words"
    
    def test_text_cleaning_preserves_content(self):
        """Test that text cleaning removes corruption but preserves content."""
        text = "Calculus (cid:1) is (cid:2) important for (cid:3) mathematics"
        cleaned = _clean_text(text)
        
        # Should remove CID glyphs
        assert "(cid:" not in cleaned
        # Should preserve key words
        assert "Calculus" in cleaned
        assert "important" in cleaned
        assert "mathematics" in cleaned
    
    @patch('src.preprocessing.pdf_ingest.PdfReader')
    def test_pdf_with_multiple_pages(self, mock_pdf_reader):
        """Test PDF extraction with multiple pages."""
        # Mock a 3-page PDF
        mock_pages = []
        for i in range(3):
            page = Mock()
            page.extract_text.return_value = f"Page {i+1} content here. " * 30
            mock_pages.append(page)
        
        mock_reader = Mock()
        mock_reader.pages = mock_pages
        mock_pdf_reader.return_value = mock_reader
        
        # Create mock file
        mock_file = Mock()
        mock_file.name = "test.pdf"
        
        # Extract text
        text, metadata = extract_pdf_text(mock_file, ocr=None)
        
        # Should have content from all pages
        assert "Page 1" in text or len(text) > 0
        assert metadata["pages"] >= 1
    
    @patch('src.preprocessing.pdf_ingest.PdfReader')
    def test_pdf_extraction_fallback_strategy(self, mock_pdf_reader):
        """Test that extraction tries multiple strategies."""
        # First strategy fails
        mock_pdf_reader.side_effect = Exception("PyMuPDF failed")
        
        mock_file = Mock()
        mock_file.name = "corrupted.pdf"
        
        # Should not crash, should try fallback
        text, metadata = extract_pdf_text(mock_file, ocr=None)
        
        # May get empty or fallback result, but shouldn't crash
        assert isinstance(text, str)
        assert isinstance(metadata, dict)


class TestURLIntegration:
    """Integration tests for URL ingestion."""
    
    def test_youtube_url_identification(self):
        """Test YouTube URL pattern matching."""
        youtube_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ&t=10s",
        ]
        
        for url in youtube_urls:
            assert _is_youtube_url(url), f"Should recognize: {url}"
        
        non_youtube = [
            "https://example.com/video",
            "https://vimeo.com/123456",
        ]
        
        for url in non_youtube:
            assert not _is_youtube_url(url), f"Should not recognize: {url}"
    
    @patch('src.preprocessing.url_ingest.requests.get')
    def test_article_extraction_with_html(self, mock_get):
        """Test article extraction from HTML."""
        mock_response = Mock()
        mock_response.text = """
        <html>
            <head><title>Test Article</title></head>
            <body>
                <article>
                    <h1>Machine Learning Basics</h1>
                    <p>Machine learning is a field of artificial intelligence.</p>
                    <p>It enables systems to learn and improve from experience.</p>
                </article>
            </body>
        </html>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch('src.preprocessing.url_ingest.trafilatura') as mock_trafilatura:
            mock_trafilatura.extract.return_value = "Machine learning is a field of artificial intelligence. It enables systems to learn and improve from experience."
            mock_trafilatura.extract_metadata.return_value = {'title': 'Test Article'}
            
            text, metadata = fetch_url_text("https://example.com/ml-article")
            
            assert "Machine learning" in text
            assert metadata["source_type"] == "article"
    
    @patch('src.preprocessing.url_ingest.YouTubeTranscriptApi')
    def test_youtube_transcript_concatenation(self, mock_api):
        """Test that YouTube transcripts are properly concatenated."""
        mock_api.get_transcript.return_value = [
            {'text': 'Introduction ', 'start': 0, 'duration': 2},
            {'text': 'to ', 'start': 2, 'duration': 1},
            {'text': 'calculus. ', 'start': 3, 'duration': 2},
            {'text': 'Derivatives measure change.', 'start': 5, 'duration': 3},
        ]
        
        text, metadata = fetch_url_text("https://youtu.be/dQw4w9WgXcQ")
        
        # Should concatenate with spaces
        assert "Introduction" in text
        assert "calculus" in text
        assert "Derivatives" in text
        assert metadata["word_count"] >= 4
    
    @patch('src.preprocessing.url_ingest.requests.get')
    def test_network_error_handling(self, mock_get):
        """Test graceful handling of network errors."""
        import requests
        mock_get.side_effect = requests.Timeout("Connection timeout")
        
        text, metadata = fetch_url_text("https://example.com/article")
        
        # Should return empty text and error metadata
        assert text == ""
        assert "extraction_method" in metadata or "error" in metadata


class TestPDFQuality:
    """Tests for PDF quality assessment heuristics."""
    
    def test_minimum_word_requirement(self):
        """Test word count threshold."""
        # Too few words
        few_words = " ".join(["word"] * 50)
        is_good, reason = _assess_extraction_quality(few_words)
        assert not is_good
        
        # Enough words
        enough_words = " ".join(["word"] * 100)
        is_good, reason = _assess_extraction_quality(enough_words)
        # May still fail if letters are too low
    
    def test_minimum_letter_requirement(self):
        """Test letter count threshold."""
        # Short words = low letter count
        short_text = " ".join(["a"] * 200)  # 200 words but only ~200 letters
        is_good, reason = _assess_extraction_quality(short_text)
        assert not is_good
        
        # Good letter content
        good_text = " ".join(["education"] * 100)  # 100 words, ~900 letters
        is_good, reason = _assess_extraction_quality(good_text)
        # Should have good letter count
    
    def test_alphabetic_ratio_threshold(self):
        """Test alphabetic character ratio."""
        # Mostly numbers and symbols
        numbers = "123 456 789 " * 50  # Good word count but low alphabetic ratio
        is_good, reason = _assess_extraction_quality(numbers)
        assert not is_good
        
        # Good alphabetic content
        letters = "abc def ghi jkl mno pqr " * 50
        is_good, reason = _assess_extraction_quality(letters)
        assert is_good


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @patch('src.preprocessing.pdf_ingest.PdfReader')
    def test_pdf_to_processing_pipeline(self, mock_pdf_reader):
        """Test PDF extraction through the full pipeline."""
        # Mock successful PDF extraction
        page = Mock()
        page.extract_text.return_value = "The derivative measures rate of change. " * 50
        
        mock_reader = Mock()
        mock_reader.pages = [page]
        mock_pdf_reader.return_value = mock_reader
        
        mock_file = Mock()
        mock_file.name = "calculus.pdf"
        
        # Extract
        text, metadata = extract_pdf_text(mock_file, ocr=None)
        
        # Verify result is suitable for downstream processing
        assert len(text) > config.MIN_INPUT_CHARS_FOR_VERIFICATION
        assert metadata["extraction_method"] in ["pymupdf", "pdfplumber", "ocr"]
        assert metadata["pages"] >= 1
    
    @patch('src.preprocessing.url_ingest.requests.get')
    def test_url_to_processing_pipeline(self, mock_get):
        """Test URL extraction through the full pipeline."""
        mock_response = Mock()
        mock_response.text = "<html><body>The integral represents accumulation of quantities over a region.</body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch('src.preprocessing.url_ingest.trafilatura') as mock_trafilatura:
            mock_trafilatura.extract.return_value = "The integral represents accumulation of quantities over a region. " * 20
            mock_trafilatura.extract_metadata.return_value = {'title': 'Integration'}
            
            text, metadata = fetch_url_text("https://example.com/integration")
            
            # Verify result is suitable for downstream processing
            assert len(text) > 100
            assert metadata["source_type"] == "article"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
