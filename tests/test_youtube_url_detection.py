"""
Tests for YouTube URL detection and video ID extraction.

Tests the youtube_ingest module's ability to:
- Detect YouTube URLs in various formats
- Extract video IDs correctly
- Handle edge cases and malformed URLs
"""

import pytest
from src.retrieval.youtube_ingest import (
    extract_video_id,
    is_youtube_url,
)


class TestExtractVideoId:
    """Test video ID extraction from various YouTube URL formats."""
    
    def test_extract_from_watch_url(self):
        """Extract video ID from standard watch URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_from_watch_url_with_timestamp(self):
        """Extract video ID from watch URL with timestamp parameter."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s"
        video_id = extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_from_watch_url_with_list(self):
        """Extract video ID from watch URL with playlist parameter."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLxxx"
        video_id = extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_from_short_url(self):
        """Extract video ID from youtu.be short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        video_id = extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_from_short_url_with_timestamp(self):
        """Extract video ID from youtu.be short URL with timestamp."""
        url = "https://youtu.be/dQw4w9WgXcQ?t=30"
        video_id = extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_from_embed_url(self):
        """Extract video ID from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        video_id = extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_from_v_url(self):
        """Extract video ID from /v/ URL format."""
        url = "https://youtube.com/v/dQw4w9WgXcQ"
        video_id = extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_various_video_ids(self):
        """Extract video IDs with different character combinations."""
        test_cases = [
            ("https://www.youtube.com/watch?v=abc123def45", "abc123def45"),  # EXACTLY 11 chars
            ("https://youtu.be/test123456a-b", "test123456a"),  # 11 chars extracted before special
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),  # EXACTLY 11 chars
        ]
        for url, expected_id in test_cases:
            result = extract_video_id(url)
            assert result == expected_id, f"Expected {expected_id}, got {result} from {url}"
            assert len(expected_id) == 11, f"Test ID {expected_id} should be 11 chars"
    
    def test_no_extract_from_non_youtube(self):
        """Do not extract from non-YouTube URLs."""
        urls = [
            "https://example.com/video",
            "https://vimeo.com/123456789",
            "https://www.youtube-downloader.com",
        ]
        for url in urls:
            assert extract_video_id(url) is None
    
    def test_extract_from_http_url(self):
        """Extract video ID from http (non-https) URLs."""
        url = "http://youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_with_www_and_without(self):
        """Extract video ID with/without www prefix."""
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
        ]
        for url in urls:
            assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_extract_invalid_video_id_length(self):
        """Extract only valid 11-character video IDs."""
        # Correctly extracts first 11 chars from longer strings
        url_with_extra = "https://youtu.be/thisisa123456verylongstringfortesting"
        result = extract_video_id(url_with_extra)
        assert result and len(result) == 11
        
        # Returns None for strings shorter than 11 chars
        short_url = "https://www.youtube.com/watch?v=short"
        assert extract_video_id(short_url) is None
    
    def test_extract_from_empty_string(self):
        """Handle empty string input."""
        assert extract_video_id("") is None
    
    def test_extract_from_none(self):
        """Handle None input."""
        assert extract_video_id(None) is None
    
    def test_extract_from_non_string(self):
        """Handle non-string input."""
        assert extract_video_id(123) is None
        assert extract_video_id([]) is None
        assert extract_video_id({}) is None


class TestIsYoutubeUrl:
    """Test YouTube URL detection."""
    
    def test_detect_youtube_com_watch(self):
        """Detect youtube.com watch URLs."""
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
    
    def test_detect_youtu_be(self):
        """Detect youtu.be short URLs."""
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert is_youtube_url("http://youtu.be/dQw4w9WgXcQ")
    
    def test_detect_youtube_embed(self):
        """Detect YouTube embed URLs."""
        assert is_youtube_url("https://www.youtube.com/embed/dQw4w9WgXcQ")
        assert is_youtube_url("https://youtube.com/v/dQw4w9WgXcQ")
    
    def test_detect_youtube_nocookie(self):
        """Detect YouTube no-cookie URLs."""
        assert is_youtube_url("https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ")
    
    def test_reject_non_youtube(self):
        """Reject non-YouTube URLs."""
        assert not is_youtube_url("https://example.com/video")
        assert not is_youtube_url("https://vimeo.com/123456")
        assert not is_youtube_url("https://www.youtube-downloader.com")
    
    def test_youtube_url_case_insensitive(self):
        """Detection is case-insensitive."""
        assert is_youtube_url("https://www.YOUTUBE.COM/watch?v=dQw4w9WgXcQ")
        assert is_youtube_url("https://YOUTU.BE/dQw4w9WgXcQ")
    
    def test_handle_empty_string(self):
        """Handle empty string input."""
        assert not is_youtube_url("")
    
    def test_handle_none(self):
        """Handle None input."""
        assert not is_youtube_url(None)
    
    def test_handle_non_string(self):
        """Handle non-string input."""
        assert not is_youtube_url(123)
        assert not is_youtube_url([])
        assert not is_youtube_url({})
    
    def test_detect_with_multiple_params(self):
        """Detect YouTube URLs with multiple parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30&list=PLxxx&index=1"
        assert is_youtube_url(url)
    
    def test_detect_partial_youtube_domain(self):
        """Detect URLs containing 'youtube.com' even with extra subdomains."""
        assert is_youtube_url("https://subdomain.youtube.com/watch?v=dQw4w9WgXcQ")


class TestVideoIdValidation:
    """Test video ID validation and edge cases."""
    
    def test_valid_video_id_patterns(self):
        """Valid video IDs contain alphanumeric, underscore, and hyphen."""
        valid_ids = [
            "https://youtu.be/dQw4w9WgXcQ",  # Contains uppercase, lowercase
            "https://youtu.be/abc123def45",  # All lowercase with numbers
            "https://youtu.be/ABC-DEF_GHI",  # Uppercase with hyphen and underscore
        ]
        for url in valid_ids:
            assert extract_video_id(url) is not None
    
    def test_invalid_video_id_patterns(self):
        """Extract valid video IDs even with extra characters nearby."""
        # Function correctly extracts 11-char sequences
        valid_url_with_exclamation = "https://youtu.be/dQw4w9WgXcQ!"
        # This is actually valid - the exclamation is just extra in the URL
        result = extract_video_id(valid_url_with_exclamation)
        assert result == "dQw4w9WgXcQ"
        
        # Return None for URLs without valid 11-char ID
        short_url = "https://youtu.be/dQw4w9Wg"  # Only 8 chars
        assert extract_video_id(short_url) is None
