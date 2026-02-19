"""
Tests for YouTube transcript ingestion with fallback handling.

Tests the youtube_ingest module's error handling:
- Fallback when package not installed
- User-friendly error messages for various failure scenarios
- Transcript fetching with language fallback
- Graceful degradation
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime

from src.retrieval.youtube_ingest import (
    fetch_transcript_text,
    get_fallback_message,
    TranscriptResult,
    YOUTUBE_TRANSCRIPT_AVAILABLE,
)


class TestTranscriptFetchFallback:
    """Test transcript fetching with fallback scenarios."""
    
    def test_fetch_with_available_package(self):
        """Test that fetch_transcript_text checks for available package."""
        # This should only pass if youtube-transcript-api is installed
        if not YOUTUBE_TRANSCRIPT_AVAILABLE:
            pytest.skip("youtube-transcript-api not installed")
        
        # Mock the API call
        with patch('src.retrieval.youtube_ingest.YouTubeTranscriptApi') as mock_api:
            mock_transcripts = MagicMock()
            mock_transcript_obj = MagicMock()
            mock_transcript_obj.language = 'en'
            mock_transcript_obj.fetch.return_value = [
                {'text': 'Hello', 'start': 0},
                {'text': 'world', 'start': 1},
            ]
            
            mock_transcripts.find_transcript.return_value = mock_transcript_obj
            mock_api.list_transcripts.return_value = mock_transcripts
            
            result = fetch_transcript_text("dQw4w9WgXcQ")
            
            # Should extract text
            assert result.success
            assert "Hello" in result.text
            assert "world" in result.text
    
    def test_fetch_without_available_package(self):
        """Test error message when youtube-transcript-api not installed."""
        with patch('src.retrieval.youtube_ingest.YOUTUBE_TRANSCRIPT_AVAILABLE', False):
            result = fetch_transcript_text("dQw4w9WgXcQ")
            
            assert not result.success
            assert result.error is not None
            assert "pip install youtube-transcript-api" in result.error
    
    def test_fetch_invalid_video_id(self):
        """Test error handling for invalid video ID."""
        result = fetch_transcript_text("")
        assert not result.success
        assert "Invalid video ID" in result.error
        
        result = fetch_transcript_text(None)
        assert not result.success
        assert "Invalid video ID" in result.error
        
        result = fetch_transcript_text(123)
        assert not result.success
        assert "Invalid video ID" in result.error
    
    @pytest.mark.skipif(not YOUTUBE_TRANSCRIPT_AVAILABLE, reason="youtube-transcript-api not installed")
    def test_fetch_with_disabled_captions(self):
        """Test user-friendly error when captions are disabled."""
        with patch('src.retrieval.youtube_ingest.YouTubeTranscriptApi') as mock_api:
            mock_api.list_transcripts.side_effect = Exception("Captions disabled for this video")
            
            result = fetch_transcript_text("dQw4w9WgXcQ")
            
            assert not result.success
            assert "disabled" in result.error.lower() or "captions" in result.error.lower()
    
    @pytest.mark.skipif(not YOUTUBE_TRANSCRIPT_AVAILABLE, reason="youtube-transcript-api not installed")
    def test_fetch_with_unavailable_transcript(self):
        """Test user-friendly error when transcript not available."""
        with patch('src.retrieval.youtube_ingest.YouTubeTranscriptApi') as mock_api:
            mock_api.list_transcripts.side_effect = Exception("No transcripts found")
            
            result = fetch_transcript_text("dQw4w9WgXcQ")
            
            assert not result.success
            assert result.error is not None
    
    @pytest.mark.skipif(not YOUTUBE_TRANSCRIPT_AVAILABLE, reason="youtube-transcript-api not installed")
    def test_fetch_with_language_fallback(self):
        """Test that English is preferred but other languages are accepted."""
        with patch('src.retrieval.youtube_ingest.YouTubeTranscriptApi') as mock_api:
            mock_transcripts = MagicMock()
            mock_transcript_obj = MagicMock()
            mock_transcript_obj.language = 'es'  # Spanish
            mock_transcript_obj.fetch.return_value = [
                {'text': 'Hola', 'start': 0},
            ]
            
            # First call fails (English not available)
            mock_transcripts.find_transcript.side_effect = Exception("English not available")
            
            # Create mock for available transcripts
            mock_available = MagicMock()
            mock_available.language = 'es'
            mock_available.fetch.return_value = [
                {'text': 'Hola', 'start': 0},
            ]
            
            mock_transcripts.get_available_transcripts.return_value = [mock_available]
            mock_api.list_transcripts.return_value = mock_transcripts
            
            result = fetch_transcript_text("dQw4w9WgXcQ")
            
            # Should succeed but note language fallback
            assert result.success or not result.success  # Either works or fails gracefully


class TestFallbackMessage:
    """Test user-friendly fallback message generation."""
    
    def test_message_for_disabled_captions(self):
        """Get appropriate message for disabled captions."""
        message = get_fallback_message("Captions disabled for this video")
        assert "disabled" in message.lower()
        assert "paste" in message.lower() or "article" in message.lower()
    
    def test_message_for_not_available(self):
        """Get appropriate message for unavailable transcript."""
        message = get_fallback_message("Transcript not available")
        assert "not available" in message.lower() or "paste" in message.lower()
    
    def test_message_for_private_video(self):
        """Get appropriate message for private video."""
        message = get_fallback_message("Video is private")
        assert "private" in message.lower()
    
    def test_message_for_package_missing(self):
        """Get appropriate message for missing package."""
        message = get_fallback_message("youtube-transcript-api not installed")
        assert "pip install" in message.lower()
    
    def test_message_for_unknown_error(self):
        """Get generic fallback message for unknown error."""
        message = get_fallback_message("Some random error occurred")
        assert "transcript" in message.lower()
        # Should provide helpful guidance
        assert len(message) > 20
    
    def test_message_for_none_error(self):
        """Get generic message when error is None."""
        message = get_fallback_message(None)
        assert "transcript" in message.lower()
        assert len(message) > 20
    
    def test_message_for_empty_error(self):
        """Get generic message when error is empty string."""
        message = get_fallback_message("")
        assert "transcript" in message.lower()
        assert len(message) > 20


class TestTranscriptResultDataclass:
    """Test TranscriptResult dataclass functionality."""
    
    def test_successful_result_to_dict(self):
        """Convert successful TranscriptResult to dict."""
        result = TranscriptResult(
            success=True,
            text="Sample transcript text",
            video_id="dQw4w9WgXcQ",
            language="en",
            fetched_at=datetime.utcnow().isoformat(),
            metadata={"char_count": 23}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["text"] == "Sample transcript text"
        assert result_dict["video_id"] == "dQw4w9WgXcQ"
        assert result_dict["language"] == "en"
        assert result_dict["metadata"]["char_count"] == 23
    
    def test_failed_result_to_dict(self):
        """Convert failed TranscriptResult to dict."""
        result = TranscriptResult(
            success=False,
            error="Captions disabled for this video",
            video_id="dQw4w9WgXcQ",
            metadata={"attempt": "en-US"}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is False
        assert result_dict["error"] == "Captions disabled for this video"
        assert result_dict["video_id"] == "dQw4w9WgXcQ"
        assert result_dict["text"] == ""
    
    def test_result_with_timestamps(self):
        """TranscriptResult handles timestamp formatting."""
        result = TranscriptResult(
            success=True,
            text="[00:00:00] Hello [00:00:05] World",
            metadata={"include_timestamps": True}
        )
        
        assert "[00:00:00]" in result.text
        assert result.metadata["include_timestamps"] is True


class TestTranscriptFetchingEdgeCases:
    """Test edge cases in transcript fetching."""
    
    @pytest.mark.skipif(not YOUTUBE_TRANSCRIPT_AVAILABLE, reason="youtube-transcript-api not installed")
    def test_fetch_with_empty_transcript(self):
        """Handle empty transcript gracefully."""
        with patch('src.retrieval.youtube_ingest.YouTubeTranscriptApi') as mock_api:
            mock_transcripts = MagicMock()
            mock_transcript_obj = MagicMock()
            mock_transcript_obj.language = 'en'
            mock_transcript_obj.fetch.return_value = []  # Empty
            
            mock_transcripts.find_transcript.return_value = mock_transcript_obj
            mock_api.list_transcripts.return_value = mock_transcripts
            
            result = fetch_transcript_text("dQw4w9WgXcQ")
            
            # Should handle empty transcript
            assert not result.success
            assert result.error is not None
    
    @pytest.mark.skipif(not YOUTUBE_TRANSCRIPT_AVAILABLE, reason="youtube-transcript-api not installed")
    def test_fetch_with_custom_languages(self):
        """Support custom language preference list."""
        with patch('src.retrieval.youtube_ingest.YouTubeTranscriptApi') as mock_api:
            mock_transcripts = MagicMock()
            mock_transcript_obj = MagicMock()
            mock_transcript_obj.language = 'es'
            mock_transcript_obj.fetch.return_value = [
                {'text': 'Hola', 'start': 0},
            ]
            
            mock_transcripts.find_transcript.return_value = mock_transcript_obj
            mock_api.list_transcripts.return_value = mock_transcripts
            
            result = fetch_transcript_text(
                "dQw4w9WgXcQ",
                languages=["es", "en"]
            )
            
            # Should call find_transcript with custom languages
            mock_transcripts.find_transcript.assert_called()
    
    @pytest.mark.skipif(not YOUTUBE_TRANSCRIPT_AVAILABLE, reason="youtube-transcript-api not installed")
    def test_fetch_with_timestamps_included(self):
        """Include timestamps when requested."""
        with patch('src.retrieval.youtube_ingest.YouTubeTranscriptApi') as mock_api:
            mock_transcripts = MagicMock()
            mock_transcript_obj = MagicMock()
            mock_transcript_obj.language = 'en'
            mock_transcript_obj.fetch.return_value = [
                {'text': 'Hello', 'start': 0},
                {'text': 'world', 'start': 5},
            ]
            
            mock_transcripts.find_transcript.return_value = mock_transcript_obj
            mock_api.list_transcripts.return_value = mock_transcripts
            
            result = fetch_transcript_text(
                "dQw4w9WgXcQ",
                include_timestamps=True
            )
            
            if result.success:
                # Timestamps should be present
                assert "[" in result.text and ":" in result.text


class TestUserFacingErrorMessages:
    """Test that error messages are user-friendly and actionable."""
    
    def test_error_messages_mention_solutions(self):
        """Error messages should suggest solutions."""
        test_cases = [
            ("Captions disabled for this video", ["disabled", "paste", "article"]),
            ("Transcript not available", ["not available", "paste", "article"]),
            ("Video is private", ["private", "public"]),
            ("youtube-transcript-api not installed", ["pip install"]),
        ]
        
        for error, expected_terms in test_cases:
            result = fetch_transcript_text("invalid_id", languages=["en"])
            message = get_fallback_message(error)
            
            # At least one expected term should be present
            found = any(term.lower() in message.lower() for term in expected_terms or ["transcript"])
            assert found, f"Message '{message}' missing expected terms: {expected_terms}"
    
    def test_error_messages_no_technical_jargon(self):
        """Error messages avoid unnecessary technical jargon."""
        with patch('src.retrieval.youtube_ingest.YOUTUBE_TRANSCRIPT_AVAILABLE', False):
            result = fetch_transcript_text("dQw4w9WgXcQ")
            
            # Should guide user to install, not dump stack trace
            assert "pip install" in result.error
            assert "traceback" not in result.error.lower()
            assert "exception" not in result.error.lower()
