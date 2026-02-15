"""
Tests for URL ingestion functionality.

Tests URL fetching, parsing, and chunking for:
- YouTube videos (transcripts)
- Web articles (HTML extraction)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.retrieval.url_ingest import (
    UrlSource,
    detect_source_type,
    extract_youtube_video_id,
    fetch_youtube_transcript,
    fetch_article_content,
    ingest_urls,
    chunk_url_sources,
    get_url_ingestion_summary
)


class TestSourceTypeDetection:
    """Test URL source type detection."""
    
    def test_detect_youtube_url(self):
        """Test YouTube URL detection."""
        youtube_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        ]
        
        for url in youtube_urls:
            assert detect_source_type(url) == "youtube", f"Failed for {url}"
    
    def test_detect_article_url(self):
        """Test article URL detection."""
        article_urls = [
            "https://example.com/article",
            "http://blog.example.org/post",
            "https://news.site/story"
        ]
        
        for url in article_urls:
            assert detect_source_type(url) == "article", f"Failed for {url}"
    
    def test_detect_unknown_url(self):
        """Test unknown URL scheme detection."""
        assert detect_source_type("ftp://example.com") == "unknown"
        assert detect_source_type("invalid-url") == "unknown"


class TestYouTubeVideoIdExtraction:
    """Test YouTube video ID extraction."""
    
    def test_extract_standard_url(self):
        """Test standard youtube.com/watch format."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_youtube_video_id(url) == "dQw4w9WgXcQ"
    
    def test_extract_short_url(self):
        """Test short youtu.be format."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_youtube_video_id(url) == "dQw4w9WgXcQ"
    
    def test_extract_embed_url(self):
        """Test embed URL format."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_youtube_video_id(url) == "dQw4w9WgXcQ"
    
    def test_extract_with_parameters(self):
        """Test URL with additional parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s"
        assert extract_youtube_video_id(url) == "dQw4w9WgXcQ"
    
    def test_invalid_url(self):
        """Test invalid URL returns None."""
        assert extract_youtube_video_id("https://youtube.com/invalid") is None


@patch('src.retrieval.url_ingest.YOUTUBE_TRANSCRIPT_AVAILABLE', True)
@patch('src.retrieval.url_ingest.YouTubeTranscriptApi')
class TestYouTubeTranscriptFetching:
    """Test YouTube transcript fetching (mocked)."""
    
    def test_fetch_transcript_success(self, mock_api):
        """Test successful transcript fetch."""
        # Mock API response
        mock_api.get_transcript.return_value = [
            {'text': 'Hello world', 'start': 0.0, 'duration': 2.0},
            {'text': 'This is a test', 'start': 2.0, 'duration': 2.5},
            {'text': 'video transcript', 'start': 4.5, 'duration': 2.0}
        ]
        
        url = "https://www.youtube.com/watch?v=test12345"
        result = fetch_youtube_transcript(url)
        
        assert result.url == url
        assert result.source_type == "youtube"
        assert result.error is None
        assert "Hello world" in result.text
        assert "This is a test" in result.text
        assert "video transcript" in result.text
        assert len(result.text) > 0
    
    def test_fetch_transcript_api_error(self, mock_api):
        """Test transcript fetch with API error."""
        mock_api.get_transcript.side_effect = Exception("Transcript not available")
        
        url = "https://www.youtube.com/watch?v=test12345"
        result = fetch_youtube_transcript(url)
        
        assert result.url == url
        assert result.source_type == "youtube"
        assert result.error is not None
        assert "Failed to fetch transcript" in result.error
        assert result.text == ""
    
    def test_fetch_transcript_invalid_video_id(self, mock_api):
        """Test transcript fetch with invalid video ID."""
        url = "https://www.youtube.com/invalid"
        result = fetch_youtube_transcript(url)
        
        assert result.error is not None
        assert "Could not extract video ID" in result.error


@patch('src.retrieval.url_ingest.REQUESTS_AVAILABLE', True)
@patch('src.retrieval.url_ingest.READABILITY_AVAILABLE', True)
@patch('src.retrieval.url_ingest.BS4_AVAILABLE', True)
@patch('src.retrieval.url_ingest.requests')
@patch('src.retrieval.url_ingest.Document')
@patch('src.retrieval.url_ingest.BeautifulSoup')
class TestArticleFetching:
    """Test web article fetching (mocked)."""
    
    def test_fetch_article_success(self, mock_bs4, mock_doc, mock_requests):
        """Test successful article fetch with readability."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.headers = {'Content-Length': '10000'}
        mock_response.iter_content = Mock(return_value=[b"<html><body>Test article content about physics.</body></html>"])
        mock_requests.get.return_value = mock_response
        
        # Mock Readability
        mock_doc_instance = Mock()
        mock_doc_instance.short_title.return_value = "Test Article"
        mock_doc_instance.summary.return_value = "<p>Test article content about physics.</p>"
        mock_doc.return_value = mock_doc_instance
        
        # Mock BeautifulSoup
        mock_soup = Mock()
        mock_soup.get_text.return_value = "Test article content about physics."
        mock_bs4.return_value = mock_soup
        
        url = "https://example.com/article"
        result = fetch_article_content(url)
        
        assert result.url == url
        assert result.source_type == "article"
        assert result.title == "Test Article"
        assert result.error is None
        assert len(result.text) > 0
        assert "physics" in result.text.lower()
    
    def test_fetch_article_too_large(self, mock_bs4, mock_doc, mock_requests):
        """Test article fetch with content too large."""
        mock_response = Mock()
        mock_response.headers = {'Content-Length': str(10 * 1024 * 1024)}  # 10MB
        mock_requests.get.return_value = mock_response
        
        url = "https://example.com/large-article"
        result = fetch_article_content(url)
        
        assert result.error is not None
        assert "too large" in result.error.lower()
        assert result.text == ""
    
    def test_fetch_article_timeout(self, mock_bs4, mock_doc, mock_requests):
        """Test article fetch with timeout."""
        import requests as real_requests
        mock_requests.get.side_effect = real_requests.exceptions.Timeout()
        mock_requests.exceptions = real_requests.exceptions
        
        url = "https://example.com/slow-article"
        result = fetch_article_content(url)
        
        assert result.error is not None
        assert "timeout" in result.error.lower()
        assert result.text == ""


class TestURLIngestion:
    """Test URL ingestion orchestration."""
    
    @patch('src.retrieval.url_ingest.fetch_youtube_transcript')
    @patch('src.retrieval.url_ingest.fetch_article_content')
    def test_ingest_mixed_urls(self, mock_article, mock_youtube):
        """Test ingesting mix of YouTube and article URLs."""
        # Mock fetchers
        mock_youtube.return_value = UrlSource(
            url="https://youtu.be/test",
            source_type="youtube",
            title="Test Video",
            fetched_at=datetime.utcnow().isoformat(),
            text="This is a video transcript with more than 200 characters to pass validation. " * 3
        )
        
        mock_article.return_value = UrlSource(
            url="https://example.com/article",
            source_type="article",
            title="Test Article",
            fetched_at=datetime.utcnow().isoformat(),
            text="This is an article with substantial content about science and education topics. " * 3
        )
        
        urls = [
            "https://youtu.be/test",
            "https://example.com/article"
        ]
        
        results = ingest_urls(urls)
        
        assert len(results) == 2
        assert results[0].source_type == "youtube"
        assert results[1].source_type == "article"
        assert all(not r.error for r in results)
    
    def test_ingest_invalid_urls(self):
        """Test ingesting invalid URLs."""
        urls = [
            "ftp://invalid.com",
            "not-a-url",
            ""
        ]
        
        results = ingest_urls(urls)
        
        # Should handle gracefully
        assert len(results) <= len(urls)
        # Invalid URLs should have errors
        for result in results:
            if result.url in urls[:2]:
                assert result.error is not None or result.text == ""


class TestURLChunking:
    """Test URL source chunking."""
    
    def test_chunk_single_source(self):
        """Test chunking a single URL source."""
        source = UrlSource(
            url="https://example.com",
            source_type="article",
            title="Test Article",
            fetched_at=datetime.utcnow().isoformat(),
            text="This is a test article. " * 50  # ~1000 chars
        )
        
        chunks = chunk_url_sources([source], chunk_size=300, overlap=50)
        
        assert len(chunks) > 0
        assert all("[Source:" in chunk for chunk in chunks)
        assert all("Test Article" in chunk for chunk in chunks)
    
    def test_chunk_multiple_sources(self):
        """Test chunking multiple URL sources."""
        sources = [
            UrlSource(
                url="https://example.com/1",
                source_type="youtube",
                title="Video 1",
                fetched_at=datetime.utcnow().isoformat(),
                text="First video transcript content. " * 30
            ),
            UrlSource(
                url="https://example.com/2",
                source_type="article",
                title="Article 2",
                fetched_at=datetime.utcnow().isoformat(),
                text="Second article text content. " * 30
            )
        ]
        
        chunks = chunk_url_sources(sources, chunk_size=400, overlap=50)
        
        assert len(chunks) > 0
        # Should have chunks from both sources
        video_chunks = [c for c in chunks if "Video 1" in c]
        article_chunks = [c for c in chunks if "Article 2" in c]
        assert len(video_chunks) > 0
        assert len(article_chunks) > 0
    
    def test_chunk_skips_errors(self):
        """Test chunking skips sources with errors."""
        sources = [
            UrlSource(
                url="https://example.com/good",
                source_type="article",
                title="Good Article",
                fetched_at=datetime.utcnow().isoformat(),
                text="Valid content. " * 50
            ),
            UrlSource(
                url="https://example.com/bad",
                source_type="article",
                title=None,
                fetched_at=datetime.utcnow().isoformat(),
                text="",
                error="Failed to fetch"
            )
        ]
        
        chunks = chunk_url_sources(sources)
        
        # Should only chunk the good source
        assert all("Good Article" in c for c in chunks)
        assert not any("bad" in c for c in chunks)


class TestURLIngestionSummary:
    """Test URL ingestion summary generation."""
    
    def test_summary_all_successful(self):
        """Test summary with all successful ingestions."""
        sources = [
            UrlSource(
                url="https://example.com/1",
                source_type="youtube",
                title="Video 1",
                fetched_at=datetime.utcnow().isoformat(),
                text="Content 1" * 100
            ),
            UrlSource(
                url="https://example.com/2",
                source_type="article",
                title="Article 2",
                fetched_at=datetime.utcnow().isoformat(),
                text="Content 2" * 100
            )
        ]
        
        summary = get_url_ingestion_summary(sources)
        
        assert summary["total_urls"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0
        assert summary["by_type"]["youtube"] == 1
        assert summary["by_type"]["article"] == 1
        assert summary["total_chars"] > 0
    
    def test_summary_with_failures(self):
        """Test summary with some failed ingestions."""
        sources = [
            UrlSource(
                url="https://example.com/good",
                source_type="article",
                title="Good",
                fetched_at=datetime.utcnow().isoformat(),
                text="Content" * 100
            ),
            UrlSource(
                url="https://example.com/bad",
                source_type="article",
                title=None,
                fetched_at=datetime.utcnow().isoformat(),
                text="",
                error="Failed"
            )
        ]
        
        summary = get_url_ingestion_summary(sources)
        
        assert summary["total_urls"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert len(summary["urls"]) == 2
        assert summary["urls"][0]["success"] is True
        assert summary["urls"][1]["success"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
