"""
Unit tests for PDF and URL ingestion modules.

Tests:
- PDF extraction quality heuristics and OCR fallback triggering
- URL ingestion with mocked HTTP and YouTube API
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

from src.preprocessing.pdf_ingest import (
    extract_pdf_text,
    _count_letters,
    _count_words,
    _compute_alphabetic_ratio,
    _assess_extraction_quality,
    _clean_text
)
from src.preprocessing.url_ingest import (
    fetch_url_text,
    _is_youtube_url,
    _extract_youtube_video_id,
    _clean_text as clean_url_text
)


class TestPDFIngestQuality:
    """Tests for PDF quality assessment."""
    
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
        assert _count_words("") == 1  # Empty string counts as 1 "word"
    
    def test_alphabetic_ratio(self):
        """Test alphabetic ratio computation."""
        assert _compute_alphabetic_ratio("hello") == 1.0
        assert _compute_alphabetic_ratio("hello123") == 5/8
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
        assert "Too few words" in reason
    
    def test_assess_extraction_quality_too_few_letters(self):
        """Test quality assessment for too few letters."""
        text = "a " * 100  # 100 words but only ~100 letters
        is_good, reason = _assess_extraction_quality(text)
        assert not is_good
        assert "Too few letters" in reason
    
    def test_assess_quality_low_alphabetic_ratio(self):
        """Test quality assessment for low alphabetic ratio."""
        text = "123 456 789 " * 50  # ~200 words but mostly numbers
        is_good, reason = _assess_extraction_quality(text)
        assert not is_good
        assert "alphabetic ratio" in reason.lower()
    
    def test_assess_quality_good_text(self):
        """Test quality assessment for good text."""
        good_text = " ".join([
            "The calculus derivative measures the rate of change of a function at a point."
            for _ in range(10)
        ])  # 150+ words with good alphabetic ratio
        is_good, reason = _assess_extraction_quality(good_text)
        assert is_good
        assert "Good quality" in reason
    
    def test_clean_text_removes_cid_glyphs(self):
        """Test that CID glyphs are removed."""
        text = "Hello (cid:1) world (cid:2)"
        cleaned = _clean_text(text)
        assert "(cid:" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned
    
    def test_clean_text_removes_excessive_whitespace(self):
        """Test that excessive whitespace is normalized."""
        text = "Hello    world  \n  test"
        cleaned = _clean_text(text)
        assert "    " not in cleaned
        assert "Hello world test" in cleaned


class TestYouTubeURLHandling:
    """Tests for YouTube URL handling."""
    
    def test_is_youtube_url(self):
        """Test YouTube URL identification."""
        assert _is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert _is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert _is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert not _is_youtube_url("https://example.com/video")
    
    def test_extract_youtube_video_id(self):
        """Test YouTube video ID extraction."""
        assert _extract_youtube_video_id("https://youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert _extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert _extract_youtube_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert _extract_youtube_video_id("https://example.com/video") == ""
    
    @patch('src.preprocessing.url_ingest.YouTubeTranscriptApi')
    def test_fetch_youtube_transcript_success(self, mock_api):
        """Test successful YouTube transcript fetching."""
        mock_api.get_transcript.return_value = [
            {'text': 'Hello ', 'start': 0, 'duration': 1},
            {'text': 'world', 'start': 1, 'duration': 1}
        ]
        
        text, metadata = fetch_url_text("https://youtube.com/watch?v=dQw4w9WgXcQ")
        
        assert "Hello" in text
        assert "world" in text
        assert metadata['source_type'] == 'youtube'
        assert metadata['video_id'] == 'dQw4w9WgXcQ'
        assert metadata['extraction_method'] == 'youtube_api'
    
    @patch('src.preprocessing.url_ingest.YouTubeTranscriptApi')
    def test_fetch_youtube_transcript_captions_disabled(self, mock_api):
        """Test YouTube transcript fetch when captions disabled."""
        mock_api.get_transcript.side_effect = Exception("Captions disabled")
        
        text, metadata = fetch_url_text("https://youtube.com/watch?v=dQw4w9WgXcQ")
        
        assert text == ""
        assert metadata['source_type'] == 'youtube'
        assert "disabled" in metadata['error'].lower() or "error" in metadata['error'].lower()


class TestArticleURLHandling:
    """Tests for article URL extraction."""
    
    @patch('src.preprocessing.url_ingest.requests.get')
    @patch('src.preprocessing.url_ingest.trafilatura')
    def test_fetch_article_with_trafilatura(self, mock_trafilatura, mock_requests):
        """Test article extraction with trafilatura."""
        mock_response = Mock()
        mock_response.text = "<html><body>Article content</body></html>"
        mock_response.raise_for_status = Mock()
        mock_requests.return_value = mock_response
        
        mock_trafilatura.extract.return_value = "Article content here with proper text"
        mock_trafilatura.extract_metadata.return_value = {'title': 'Test Article'}
        
        text, metadata = fetch_url_text("https://example.com/article")
        
        assert "Article content" in text
        assert metadata['source_type'] == 'article'
        assert metadata['extraction_method'] == 'trafilatura'
    
    @patch('src.preprocessing.url_ingest.requests.get')
    def test_fetch_article_request_timeout(self, mock_requests):
        """Test article fetch with request timeout."""
        mock_requests.side_effect = Exception("Connection timeout")
        
        text, metadata = fetch_url_text("https://example.com/article")
        
        assert text == ""
        assert metadata['extraction_method'] == 'error'
        assert "timeout" in metadata['error'].lower() or "error" in metadata['error'].lower()
    
    @patch('src.preprocessing.url_ingest.requests.get')
    def test_fetch_article_empty_extraction(self, mock_requests):
        """Test article with empty text extraction."""
        mock_response = Mock()
        mock_response.text = "<html><body></body></html>"
        mock_response.raise_for_status = Mock()
        mock_requests.return_value = mock_response
        
        with patch('src.preprocessing.url_ingest.trafilatura') as mock_trafilatura:
            mock_trafilatura.extract.return_value = None
            
            text, metadata = fetch_url_text("https://example.com/article")
            
            # Should try fallback strategies
            assert metadata['extraction_method'] is not None


class TestTextCleaning:
    """Tests for text cleaning utilities."""
    
    def test_clean_url_text(self):
        """Test URL text cleaning."""
        text = "Hello (cid:123) world   with  spaces"
        cleaned = clean_url_text(text)
        
        assert "(cid:" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned
    
    def test_clean_text_preserves_content(self):
        """Test that cleaning doesn't remove valid content."""
        text = "Python is great! Numbers: 123, symbols: @#$"
        cleaned = _clean_text(text)
        
        assert "Python" in cleaned
        assert "great" in cleaned


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
