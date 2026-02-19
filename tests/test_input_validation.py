"""
Tests for input validation helpers.

Tests URL parsing, validation, and input readiness checking.
"""

import pytest
from src.ui.input_validation import (
    parse_and_validate_urls,
    has_any_input,
    get_input_status_message,
    validate_urls_for_processing,
    is_youtube_url,
    extract_youtube_video_id,
)


class TestParseAndValidateURLs:
    """Tests for URL parsing and validation."""

    def test_parse_single_valid_url(self) -> None:
        """Test parsing a single valid HTTPS URL."""
        urls_text = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        valid, invalid = parse_and_validate_urls(urls_text)
        
        assert len(valid) == 1
        assert valid[0] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert len(invalid) == 0

    def test_parse_multiple_valid_urls(self) -> None:
        """Test parsing multiple valid URLs."""
        urls_text = """https://www.youtube.com/watch?v=dQw4w9WgXcQ
http://example.com/article
https://docs.python.org"""
        
        valid, invalid = parse_and_validate_urls(urls_text)
        
        assert len(valid) == 3
        assert len(invalid) == 0
        assert "https://www.youtube.com/watch?v=dQw4w9WgXcQ" in valid
        assert "http://example.com/article" in valid
        assert "https://docs.python.org" in valid

    def test_parse_with_whitespace_trimming(self) -> None:
        """Test that whitespace is properly trimmed."""
        urls_text = """  https://example.com  
        
http://other.com  """
        
        valid, invalid = parse_and_validate_urls(urls_text)
        
        assert len(valid) == 2
        assert "https://example.com" in valid
        assert "http://other.com" in valid
        # Verify no leading/trailing spaces
        for url in valid:
            assert not url.startswith(" ")
            assert not url.endswith(" ")

    def test_parse_with_invalid_urls(self) -> None:
        """Test that non-http(s) URLs are marked as invalid."""
        urls_text = """https://valid.com
ftp://invalid.com
not-a-url
example.com"""
        
        valid, invalid = parse_and_validate_urls(urls_text)
        
        assert len(valid) == 1
        assert valid[0] == "https://valid.com"
        assert len(invalid) == 3
        assert "ftp://invalid.com" in invalid
        assert "not-a-url" in invalid
        assert "example.com" in invalid

    def test_parse_with_comments_and_empty_lines(self) -> None:
        """Test that comments and empty lines are skipped."""
        urls_text = """# This is a comment
https://valid.com

# Another comment
http://example.com
"""
        
        valid, invalid = parse_and_validate_urls(urls_text)
        
        assert len(valid) == 2
        assert len(invalid) == 0
        assert "https://valid.com" in valid
        assert "http://example.com" in valid

    def test_parse_empty_string(self) -> None:
        """Test parsing empty string."""
        valid, invalid = parse_and_validate_urls("")
        
        assert len(valid) == 0
        assert len(invalid) == 0

    def test_parse_whitespace_only(self) -> None:
        """Test parsing whitespace-only string."""
        valid, invalid = parse_and_validate_urls("   \n\n   ")
        
        assert len(valid) == 0
        assert len(invalid) == 0


class TestHasAnyInput:
    """Tests for input readiness checking."""

    def test_has_any_input_with_text_only(self) -> None:
        """Test with text input only."""
        result = has_any_input(notes_text="Some notes here", min_text_chars=1)
        assert result is True

    def test_has_any_input_with_files_only(self) -> None:
        """Test with files only."""
        result = has_any_input(notes_images=["file1", "file2"])
        assert result is True

    def test_has_any_input_with_audio_only(self) -> None:
        """Test with audio file only."""
        mock_audio = object()
        result = has_any_input(audio_file=mock_audio)
        assert result is True

    def test_has_any_input_with_urls_only(self) -> None:
        """Test with valid URLs only.
        
        This is a key test case - URLs should enable buttons.
        """
        urls_text = """https://www.youtube.com/watch?v=dQw4w9WgXcQ
https://example.com/article"""
        
        result = has_any_input(urls_text=urls_text)
        assert result is True

    def test_has_any_input_with_invalid_urls_only(self) -> None:
        """Test with invalid URLs only.
        
        Invalid URLs should NOT count as input.
        """
        urls_text = """ftp://invalid.com
not-a-url
example.com"""
        
        result = has_any_input(urls_text=urls_text)
        assert result is False

    def test_has_any_input_with_no_input(self) -> None:
        """Test with no input."""
        result = has_any_input(
            notes_text="",
            notes_images=[],
            audio_file=None,
            urls_text=""
        )
        assert result is False

    def test_has_any_input_with_empty_text(self) -> None:
        """Test that empty or whitespace-only text doesn't count."""
        result = has_any_input(notes_text="   \n  ")
        assert result is False

    def test_has_any_input_with_mixed_valid_invalid_urls(self) -> None:
        """Test with mix of valid and invalid URLs.
        
        Should be True because valid URLs exist.
        """
        urls_text = """https://valid.com
invalid-url
http://another-valid.com"""
        
        result = has_any_input(urls_text=urls_text)
        assert result is True

    def test_has_any_input_with_min_text_chars(self) -> None:
        """Test minimum text character requirement."""
        # Too short
        result = has_any_input(notes_text="ab", min_text_chars=10)
        assert result is False
        
        # Exactly minimum
        result = has_any_input(notes_text="0123456789", min_text_chars=10)
        assert result is True
        
        # Minimum with whitespace (trimmed)
        result = has_any_input(notes_text="  ab  ", min_text_chars=10)
        assert result is False


class TestGetInputStatusMessage:
    """Tests for input status messaging."""

    def test_no_input_message(self) -> None:
        """Test message when no input provided."""
        msg = get_input_status_message(
            notes_text="",
            notes_images=[],
            audio_file=None,
            urls_text=""
        )
        assert msg == "Upload a file, paste text, or add URLs."

    def test_invalid_urls_only_message(self) -> None:
        """Test message when only invalid URLs provided."""
        msg = get_input_status_message(
            notes_text="",
            notes_images=[],
            audio_file=None,
            urls_text="invalid-url\nftp://bad.com"
        )
        assert "Invalid URL format" in msg
        assert "https://" in msg or "http://" in msg

    def test_valid_input_no_message(self) -> None:
        """Test that no message is shown when valid input exists."""
        msg = get_input_status_message(
            notes_text="Some notes",
            urls_text=""
        )
        assert msg == ""

    def test_valid_urls_no_message(self) -> None:
        """Test that no message is shown with valid URLs."""
        msg = get_input_status_message(
            notes_text="",
            notes_images=[],
            audio_file=None,
            urls_text="https://example.com"
        )
        assert msg == ""


class TestValidateURLsForProcessing:
    """Tests for URL validation for pipeline processing."""

    def test_validate_valid_urls(self) -> None:
        """Test validation of valid URLs."""
        urls_text = """https://example.com/article
http://other.com/resource"""
        
        urls, error = validate_urls_for_processing(urls_text)
        
        assert len(urls) == 2
        assert error is None
        assert "https://example.com/article" in urls
        assert "http://other.com/resource" in urls

    def test_validate_youtube_url_accepted(self) -> None:
        """Test that YouTube URLs are now accepted (content extraction in pipeline).
        
        YouTube URLs are now processed via youtube-transcript-api in the pipeline.
        """
        urls_text = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        urls, error = validate_urls_for_processing(urls_text)
        
        assert len(urls) == 1
        assert error is None
        assert "https://www.youtube.com/watch?v=dQw4w9WgXcQ" in urls

    def test_validate_youtu_be_url_accepted(self) -> None:
        """Test that shortened YouTube URLs (youtu.be) are also accepted."""
        urls_text = "https://youtu.be/xyz123"
        
        urls, error = validate_urls_for_processing(urls_text)
        
        assert len(urls) == 1
        assert error is None
        assert "https://youtu.be/xyz123" in urls

    def test_validate_mixed_youtube_and_articles(self) -> None:
        """Test when both YouTube and article URLs provided.
        
        Both are now accepted for processing.
        """
        urls_text = """https://example.com/article
https://www.youtube.com/watch?v=xyz"""
        
        urls, error = validate_urls_for_processing(urls_text)
        
        assert len(urls) == 2
        assert error is None

    def test_validate_article_urls_accepted(self) -> None:
        """Test that article URLs (non-YouTube) are accepted."""
        urls_text = """https://wikipedia.org/wiki/Python
https://medium.com/@author/article
http://documentation.com"""
        
        urls, error = validate_urls_for_processing(urls_text)
        
        assert len(urls) == 3
        assert error is None

    def test_validate_mixed_urls(self) -> None:
        """Test validation with mix of valid/invalid.
        
        Should return only valid URLs (non-YouTube).
        """
        urls_text = """https://valid.com
invalid-url
http://other-valid.com"""
        
        urls, error = validate_urls_for_processing(urls_text)
        
        assert len(urls) == 2
        assert error is None
        assert "https://valid.com" in urls
        assert "http://other-valid.com" in urls
        assert "invalid-url" not in urls

    def test_validate_only_invalid_urls(self) -> None:
        """Test validation with only invalid URLs."""
        urls_text = """ftp://notallowed.com
example.com
not-a-url"""
        
        urls, error = validate_urls_for_processing(urls_text)
        
        assert len(urls) == 0
        assert error is not None
        assert "Invalid URL format" in error or "invalid" in error.lower()

    def test_validate_empty_urls(self) -> None:
        """Test validation with empty URLs."""
        urls, error = validate_urls_for_processing("")
        
        assert len(urls) == 0
        assert error is None

    def test_validate_none_urls(self) -> None:
        """Test validation with None URLs."""
        urls, error = validate_urls_for_processing(None)
        
        assert len(urls) == 0
        assert error is None


class TestYouTubeDetection:
    """Tests for YouTube URL detection."""

    def test_is_youtube_url_with_full_url(self) -> None:
        """Test YouTube detection with full URL."""
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_is_youtube_url_with_shortened(self) -> None:
        """Test YouTube detection with shortened youtu.be URL."""
        assert is_youtube_url("https://youtu.be/xyz123") is True

    def test_is_youtube_url_case_insensitive(self) -> None:
        """Test YouTube detection is case-insensitive."""
        assert is_youtube_url("https://www.YOUTUBE.com/watch?v=xyz") is True
        assert is_youtube_url("https://YOUTU.be/xyz") is True

    def test_is_youtube_url_with_non_youtube(self) -> None:
        """Test non-YouTube URLs return False."""
        assert is_youtube_url("https://example.com") is False
        assert is_youtube_url("https://vimeo.com/video") is False

    def test_extract_youtube_video_id_from_full_url(self) -> None:
        """Test extracting video ID from full YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = extract_youtube_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_youtube_video_id_from_shortened(self) -> None:
        """Test extracting video ID from shortened URL."""
        url = "https://youtu.be/xyz123ABC"
        video_id = extract_youtube_video_id(url)
        assert video_id == "xyz123ABC"

    def test_extract_youtube_video_id_invalid(self) -> None:
        """Test extracting ID from non-YouTube URL."""
        url = "https://example.com/video"
        video_id = extract_youtube_video_id(url)
        assert video_id is None


class TestValidationAllowsURLsOnly:
    """Tests for backend validation accepting URLs-only input."""

    def test_validation_allows_urls_only(self) -> None:
        """Test that backend validation accepts URLs-only input.
        
        This is a key backend validation test - URLs alone should pass
        (excluding YouTube which requires transcript).
        """
        urls_text = """https://wikipedia.org/wiki/Machine_Learning
https://medium.com/@author/understanding-ml"""
        
        urls, error = validate_urls_for_processing(urls_text)
        
        # Should accept these URLs
        assert len(urls) == 2
        assert error is None
        assert "Please provide" not in str(error or "")

    def test_validation_rejects_invalid_urls_only(self) -> None:
        """Test that backend validation rejects invalid-URLs-only input.
        
        Backend should show error for invalid URLs with guidance.
        """
        urls_text = "not-a-url\ninvalid-format"
        
        urls, error = validate_urls_for_processing(urls_text)
        
        # Should reject and provide error
        assert len(urls) == 0
        assert error is not None
        assert "invalid URL" in error.lower() or "must start with http" in error.lower()

    def test_validation_accepts_youtube_urls(self) -> None:
        """Test that YouTube URLs are now accepted for processing.
        
        Transcript extraction happens in the processing pipeline,
        not at validation time.
        """
        urls_text = "https://www.youtube.com/watch?v=video123"
        
        urls, error = validate_urls_for_processing(urls_text)
        
        # YouTube URLs are now accepted
        assert len(urls) == 1
        assert error is None
        assert "https://www.youtube.com/watch?v=video123" in urls


class TestIntegrationScenarios:
    """Integration tests for realistic user scenarios."""

    def test_user_enters_only_article_url(self) -> None:
        """Scenario: User adds only article URL, should work."""
        urls_text = "https://example.com/article"
        
        # Check input readiness
        has_input = has_any_input(urls_text=urls_text)
        assert has_input is True
        
        # Get status message (should be empty because input exists)
        status_msg = get_input_status_message(urls_text=urls_text)
        assert status_msg == ""
        
        # Get URLs for processing
        urls, error = validate_urls_for_processing(urls_text)
        assert len(urls) == 1
        assert error is None

    def test_user_enters_only_youtube_url(self) -> None:
        """Scenario: User adds only YouTube URL, transcript will be extracted.
        
        Key scenario: YouTube URLs are now accepted and will be processed
        in the pipeline to extract transcripts.
        """
        urls_text = "https://www.youtube.com/watch?v=xyz123"
        
        # Check input readiness (buttons enabled - URL is present)
        has_input = has_any_input(urls_text=urls_text)
        assert has_input is True
        
        # Get URLs for processing (should now succeed)
        urls, error = validate_urls_for_processing(urls_text)
        assert len(urls) == 1
        assert error is None
        assert "https://www.youtube.com/watch?v=xyz123" in urls

    def test_user_enters_invalid_youtube_url(self) -> None:
        """Scenario: User enters malformed YouTube URL."""
        urls_text = "youtube.com/watch?v=xyz123"  # Missing protocol
        
        # Check input readiness
        has_input = has_any_input(urls_text=urls_text)
        assert has_input is False
        
        # Get status message
        status_msg = get_input_status_message(urls_text=urls_text)
        assert "Invalid URL format" in status_msg

    def test_user_enters_text_and_url(self) -> None:
        """Scenario: User provides both text and URL."""
        notes_text = "Lesson notes"
        urls_text = "https://example.com/resource"
        
        has_input = has_any_input(notes_text=notes_text, urls_text=urls_text)
        assert has_input is True
        
        # Both should be available for processing
        urls, error = validate_urls_for_processing(urls_text)
        assert len(urls) == 1
        assert error is None

    def test_user_enters_only_invalid_urls(self) -> None:
        """Scenario: User enters only invalid URLs, should show message."""
        urls_text = "notaurl.com\nftp://invalid.ftp"
        
        has_input = has_any_input(urls_text=urls_text)
        assert has_input is False
        
        status_msg = get_input_status_message(urls_text=urls_text)
        assert "Invalid URL format" in status_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
