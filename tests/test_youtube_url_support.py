"""
Comprehensive tests for YouTube URL support in the evidence ingestion system.

Tests:
- YouTube URL detection and video ID extraction
- Transcript fetching with mocked API responses
- Optional timestamp formatting
- Language preference (English, then fallback)
- Integration with Evidence Store and EvidenceStoreBuilder
- Error handling for various failure scenarios
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from typing import List, Dict, Any

from src.retrieval.url_ingest import (
    UrlSource,
    detect_source_type,
    extract_youtube_video_id,
    fetch_youtube_transcript,
    ingest_urls,
    chunk_url_sources,
    get_url_ingestion_summary,
    YOUTUBE_INCLUDE_TIMESTAMPS
)
from src.retrieval.evidence_builder import build_session_evidence_store, add_url_sources_to_store
from src.retrieval.evidence_store import Evidence, EvidenceStore


class TestYouTubeURLDetection:
    """Test YouTube URL detection and validation."""
    
    def test_detect_youtube_with_watch(self):
        """Detect standard YouTube watch URLs."""
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ&t=10s",
        ]
        for url in urls:
            assert detect_source_type(url) == "youtube", f"Failed for {url}"
    
    def test_detect_youtube_short_urls(self):
        """Detect short youtu.be URLs."""
        urls = [
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?t=10",
            "http://youtu.be/dQw4w9WgXcQ",
        ]
        for url in urls:
            assert detect_source_type(url) == "youtube", f"Failed for {url}"
    
    def test_detect_youtube_embed_urls(self):
        """Detect YouTube embed URLs."""
        urls = [
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://youtube.com/v/dQw4w9WgXcQ",
        ]
        for url in urls:
            assert detect_source_type(url) == "youtube", f"Failed for {url}"
    
    def test_detect_non_youtube_urls(self):
        """Ensure non-YouTube URLs are not detected as YouTube."""
        urls = [
            "https://example.com/video",
            "https://vimeo.com/123456",
            "https://www.youtube-dl.example.com",
        ]
        for url in urls:
            assert detect_source_type(url) != "youtube", f"Incorrectly detected: {url}"
    
    def test_extract_video_id_watch(self):
        """Extract video ID from watch URLs."""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=abc123def45", "abc123def45"),
            ("https://www.youtube.com/watch?v=test123456A&t=10s", "test123456A"),
        ]
        for url, expected_id in test_cases:
            assert extract_youtube_video_id(url) == expected_id
    
    def test_extract_video_id_short(self):
        """Extract video ID from short URLs."""
        test_cases = [
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/abc123def45", "abc123def45"),
            ("https://youtu.be/test123456a?t=5", "test123456a"),
        ]
        for url, expected_id in test_cases:
            assert extract_youtube_video_id(url) == expected_id
    
    def test_extract_video_id_embed(self):
        """Extract video ID from embed URLs."""
        test_cases = [
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/v/abc123def45", "abc123def45"),
        ]
        for url, expected_id in test_cases:
            assert extract_youtube_video_id(url) == expected_id
    
    def test_extract_invalid_video_id(self):
        """Return None for invalid URLs."""
        urls = [
            "https://example.com/video",
            "https://youtube.com/",
            "not-a-url",
        ]
        for url in urls:
            assert extract_youtube_video_id(url) is None


@patch('src.retrieval.youtube_ingest.YOUTUBE_TRANSCRIPT_AVAILABLE', True)
@patch('src.retrieval.youtube_ingest.YouTubeTranscriptApi')
class TestYouTubeTranscriptFetching:
    """Test YouTube transcript fetching with mocked API."""
    
    def test_fetch_successful_transcript(self, mock_api):
        """Successfully fetch and format transcript."""
        mock_api.list_transcripts.return_value.find_transcript.return_value.fetch.return_value = [
            {'text': 'Hello', 'start': 0.0, 'duration': 1.0},
            {'text': 'world', 'start': 1.0, 'duration': 1.0},
            {'text': 'Welcome to', 'start': 2.0, 'duration': 1.5},
            {'text': 'the video', 'start': 3.5, 'duration': 1.0},
        ]
        
        url = "https://www.youtube.com/watch?v=test123456ab"
        result = fetch_youtube_transcript(url)
        
        assert result.url == url
        assert result.source_type == "youtube"
        assert result.error is None
        assert "Hello" in result.text
        assert "world" in result.text
        assert len(result.text) > 0
    
    def test_fetch_transcript_with_timestamps(self, mock_api):
        """Fetch transcript with timestamps included."""
        mock_api.list_transcripts.return_value.find_transcript.return_value.fetch.return_value = [
            {'text': 'Introduction', 'start': 0.0, 'duration': 2.0},
            {'text': 'Main content', 'start': 2.0, 'duration': 30.0},
            {'text': 'Conclusion', 'start': 32.0, 'duration': 2.0},
        ]
        
        url = "https://youtu.be/test123456ab"
        result = fetch_youtube_transcript(url, include_timestamps=True)
        
        assert result.error is None
        assert "[00:00:00]" in result.text
        assert "[00:00:02]" in result.text
        assert "[00:00:32]" in result.text
        assert "Introduction" in result.text
        assert "Main content" in result.text
        assert "Conclusion" in result.text
    
    def test_fetch_transcript_without_timestamps(self, mock_api):
        """Fetch transcript without timestamps."""
        mock_api.list_transcripts.return_value.find_transcript.return_value.fetch.return_value = [
            {'text': 'Part A', 'start': 0.0, 'duration': 2.0},
            {'text': 'Part B', 'start': 2.0, 'duration': 3.0},
        ]
        
        url = "https://www.youtube.com/watch?v=test123456ab"
        result = fetch_youtube_transcript(url, include_timestamps=False)
        
        assert result.error is None
        assert "Part A" in result.text
        assert "Part B" in result.text
        assert "[00:" not in result.text
    
    def test_fetch_transcript_english_preferred(self, mock_api):
        """Prefer English transcripts when available."""
        mock_transcripts = MagicMock()
        mock_api.list_transcripts.return_value = mock_transcripts
        mock_transcripts.find_transcript.return_value.fetch.return_value = [
            {'text': 'English transcript', 'start': 0, 'duration': 1}
        ]
        
        url = "https://youtu.be/test123456ab"
        result = fetch_youtube_transcript(url)
        
        assert result.error is None
        assert "English transcript" in result.text
        mock_transcripts.find_transcript.assert_called_once_with(['en', 'en-US'])
    
    def test_fetch_transcript_language_fallback(self, mock_api):
        """Fall back to available transcript when English unavailable."""
        mock_transcripts = MagicMock()
        mock_api.list_transcripts.return_value = mock_transcripts
        
        mock_transcripts.find_transcript.side_effect = Exception("English not available")
        mock_transcripts.find_manually_created_transcript.side_effect = Exception("No manual")
        
        mock_available = MagicMock()
        mock_available.language = "es"
        mock_available.fetch.return_value = [
            {'text': 'Spanish transcript', 'start': 0, 'duration': 1}
        ]
        mock_transcripts.get_available_transcripts.return_value = [mock_available]
        
        url = "https://youtu.be/test123456ab"
        result = fetch_youtube_transcript(url)
        
        assert result.error is None
        assert "Spanish transcript" in result.text
    
    def test_fetch_transcript_invalid_video_id(self, mock_api):
        """Handle invalid video ID gracefully."""
        url = "https://www.youtube.com/watch?v=invalid"
        result = fetch_youtube_transcript(url)
        
        assert result.error is not None
        assert "Could not extract video ID" in result.error
        assert result.text == ""
        assert result.source_type == "youtube"
    
    def test_fetch_transcript_api_error(self, mock_api):
        """Handle YouTube API errors gracefully with user-friendly messages."""
        mock_api.list_transcripts.side_effect = Exception("API Error: Video not found")
        
        url = "https://www.youtube.com/watch?v=test123456ab"
        result = fetch_youtube_transcript(url)
        
        assert result.error is not None
        # Error message should be user-friendly (not "Failed to fetch transcript")
        assert ("Transcript" in result.error or "Video" in result.error or "Unable" in result.error)
        assert result.text == ""
        assert result.source_type == "youtube"
    
    def test_fetch_transcript_metadata(self, mock_api):
        """Verify metadata includes video_id and language."""
        mock_api.list_transcripts.return_value.find_transcript.return_value.fetch.return_value = [
            {'text': 'Test content', 'start': 0, 'duration': 1}
        ]
        mock_api.list_transcripts.return_value.find_transcript.return_value.language = 'en'
        
        url = "https://youtu.be/dQw4w9WgXcQ"
        result = fetch_youtube_transcript(url)
        
        assert result.metadata is not None
        assert result.metadata['video_id'] == 'dQw4w9WgXcQ'
        assert result.metadata['include_timestamps'] == False


@patch('src.retrieval.url_ingest.YOUTUBE_INCLUDE_TIMESTAMPS', False)
@patch('src.retrieval.youtube_ingest.YOUTUBE_TRANSCRIPT_AVAILABLE', True)
@patch('src.retrieval.url_ingest.REQUESTS_AVAILABLE', True)
@patch('src.retrieval.youtube_ingest.YouTubeTranscriptApi')
@patch('src.retrieval.url_ingest.requests.get')
class TestURLIngestionWithYouTube:
    """Test URL ingestion orchestration with YouTube support."""
    
    def test_ingest_single_youtube_url(self, mock_requests, mock_api):
        """Ingest single YouTube URL."""
        mock_api.list_transcripts.return_value.find_transcript.return_value.fetch.return_value = [
            {'text': 'Test transcript', 'start': 0, 'duration': 1}
        ]
        
        urls = ["https://youtu.be/test123456ab"]
        results = ingest_urls(urls)
        
        assert len(results) == 1
        assert results[0].source_type == "youtube"
        assert results[0].error is None
        assert "Test transcript" in results[0].text
    
    def test_ingest_mixed_urls(self, mock_requests, mock_api):
        """Ingest mix of YouTube and article URLs."""
        mock_api.list_transcripts.return_value.find_transcript.return_value.fetch.return_value = [
            {'text': 'Video content', 'start': 0, 'duration': 1}
        ]
        
        mock_response = MagicMock()
        mock_response.headers = {'Content-Length': '5000'}
        mock_response.iter_content.return_value = [
            b'<html><body><p>Article content here</p></body></html>'
        ]
        mock_requests.return_value = mock_response
        
        urls = [
            "https://youtu.be/test123456ab",
            "https://example.com/article"
        ]
        results = ingest_urls(urls)
        
        assert len(results) == 2
        assert results[0].source_type == "youtube"
        assert results[1].source_type == "article"
    
    def test_ingest_with_timestamps_enabled(self, mock_requests, mock_api):
        """Ingest YouTube with timestamps enabled globally."""
        mock_api.list_transcripts.return_value.find_transcript.return_value.fetch.return_value = [
            {'text': 'First part', 'start': 0.0, 'duration': 1.0},
            {'text': 'Second part', 'start': 1.0, 'duration': 1.0},
        ]
        
        urls = ["https://youtu.be/test123456ab"]
        results = ingest_urls(urls, include_timestamps=True)
        
        assert len(results) == 1
        assert "[00:00:00]" in results[0].text
        assert "[00:00:01]" in results[0].text


class TestYouTubeIntegrationWithEvidenceStore:
    """Test integration of YouTube URLs with Evidence Store."""
    
    @patch('src.retrieval.youtube_ingest.YOUTUBE_TRANSCRIPT_AVAILABLE', True)
    @patch('src.retrieval.youtube_ingest.YouTubeTranscriptApi')
    def test_youtube_url_in_evidence_builder(self, mock_api):
        """YouTube URLs should be included in evidence store building."""
        mock_api.list_transcripts.return_value.find_transcript.return_value.fetch.return_value = [
            {'text': 'YouTube evidence', 'start': 0, 'duration': 1}
        ]
        
        session_id = "test_session"
        input_text = "Main input text " * 100
        urls = ["https://youtu.be/test123456ab"]
        
        store, stats = build_session_evidence_store(
            session_id=session_id,
            input_text=input_text,
            urls=urls
        )
        
        assert store is not None
        assert len(store.evidence) > 0
        assert stats['url_ingestion'] is not None
        assert stats['url_ingestion']['successful'] >= 1
    
    @patch('src.retrieval.youtube_ingest.YOUTUBE_TRANSCRIPT_AVAILABLE', True)
    @patch('src.retrieval.youtube_ingest.YouTubeTranscriptApi')
    def test_youtube_evidence_metadata(self, mock_api):
        """YouTube evidence should have correct source type and ID."""
        long_text = "Video content " * 15
        mock_api.list_transcripts.return_value.find_transcript.return_value.fetch.return_value = [
            {'text': long_text, 'start': 0, 'duration': 1}
        ]
        
        session_id = "test_session"
        input_text = "Main input " * 100
        youtube_url = "https://youtu.be/test123456ab"
        urls = [youtube_url]
        
        store, stats = build_session_evidence_store(
            session_id=session_id,
            input_text=input_text,
            urls=urls
        )
        
        youtube_evidence = [ev for ev in store.evidence if ev.source_type == "youtube"]
        
        assert len(youtube_evidence) > 0
        
        for ev in youtube_evidence:
            assert ev.source_type == "youtube"
            assert ev.source_id == youtube_url
            assert len(ev.text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
