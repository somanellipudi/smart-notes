"""
Input validation helpers for Streamlit UI.

Centralizes logic for checking input readiness and parsing/validating URLs.
"""

import logging
import re
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def is_youtube_url(url: str) -> bool:
    """
    Check if a URL is a YouTube link.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL is from youtube.com or youtu.be
    """
    return (
        "youtube.com" in url.lower() or
        "youtu.be" in url.lower()
    )


def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from a YouTube URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID if found, None otherwise
    """
    # Match youtube.com/watch?v=VIDEO_ID
    watch_match = re.search(r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)', url)
    if watch_match:
        return watch_match.group(1)
    
    # Match youtu.be/VIDEO_ID
    short_match = re.search(r'youtu\.be/([a-zA-Z0-9_-]+)', url)
    if short_match:
        return short_match.group(1)
    
    return None


def parse_and_validate_urls(urls_text: str) -> Tuple[List[str], List[str]]:
    """
    Parse URL text (one per line) and validate format.
    
    Args:
        urls_text: Text with URLs separated by newlines
        
    Returns:
        Tuple of (valid_urls, invalid_urls)
        - valid_urls: List of URLs that start with http:// or https://
        - invalid_urls: List of non-empty URLs that don't match http(s) format
    """
    if not urls_text or not urls_text.strip():
        return [], []
    
    valid_urls = []
    invalid_urls = []
    
    for line in urls_text.split('\n'):
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Check if valid URL format
        if line.startswith('http://') or line.startswith('https://'):
            valid_urls.append(line)
        else:
            invalid_urls.append(line)
    
    return valid_urls, invalid_urls


def has_any_input(
    notes_text: Optional[str] = None,
    notes_images: Optional[List] = None,
    audio_file: Optional[object] = None,
    urls_text: Optional[str] = None,
    min_text_chars: int = 1
) -> bool:
    """
    Check if user has provided any valid input.
    
    Input is considered valid if ANY of:
    - notes_text has at least min_text_chars characters (trimmed)
    - notes_images has at least one file
    - audio_file is provided
    - urls_text has at least one valid URL
    
    Args:
        notes_text: User-typed or pasted notes
        notes_images: List of uploaded files
        audio_file: Uploaded audio file object
        urls_text: URLs entered in text area (newline-separated)
        min_text_chars: Minimum characters for notes_text to count as valid
        
    Returns:
        True if any input is present, False otherwise
    """
    # Check text input
    if notes_text and len(notes_text.strip()) >= min_text_chars:
        return True
    
    # Check files
    if notes_images and len(notes_images) > 0:
        return True
    
    # Check audio
    if audio_file is not None:
        return True
    
    # Check URLs
    if urls_text:
        valid_urls, _ = parse_and_validate_urls(urls_text)
        if len(valid_urls) > 0:
            return True
    
    return False


def get_input_status_message(
    notes_text: Optional[str] = None,
    notes_images: Optional[List] = None,
    audio_file: Optional[object] = None,
    urls_text: Optional[str] = None,
    min_text_chars: int = 1
) -> str:
    """
    Get a user-friendly message about input status.
    
    Returns appropriate message based on what input is present/missing.
    """
    # Check each input type
    has_text = notes_text and len(notes_text.strip()) >= min_text_chars
    has_files = notes_images and len(notes_images) > 0
    has_audio = audio_file is not None
    
    valid_urls = []
    invalid_urls = []
    if urls_text:
        valid_urls, invalid_urls = parse_and_validate_urls(urls_text)
    
    has_valid_urls = len(valid_urls) > 0
    has_invalid_urls = len(invalid_urls) > 0
    
    # Determine message based on what's available
    if not (has_text or has_files or has_audio or has_valid_urls or has_invalid_urls):
        return "Upload a file, paste text, or add URLs."
    
    if has_invalid_urls and not has_valid_urls:
        return "Invalid URL format (must start with https:// or http://)."
    
    # All other cases have at least some valid input
    return ""


def validate_urls_for_processing(urls_text: Optional[str]) -> Tuple[List[str], Optional[str]]:
    """
    Validate URLs for processing and return along with error message if any.
    
    Special handling:
    - YouTube URLs are allowed (transcript extraction via youtube-transcript-api)
    - Article URLs are allowed (text extraction)
    - Invalid URL formats result in error
    
    Args:
        urls_text: URLs text from UI
        
    Returns:
        Tuple of (urls_list, error_message)
        - urls_list: List of valid URLs ready for processing, empty list if no URLs provided
        - error_message: User-friendly error message if invalid URLs found, None if all OK
    """
    if not urls_text or not urls_text.strip():
        return [], None
    
    valid_urls, invalid_urls = parse_and_validate_urls(urls_text)
    
    if not valid_urls and invalid_urls:
        # Only invalid URLs were provided
        error_message = (
            f"Found {len(invalid_urls)} invalid URL(s). "
            "URLs must start with http:// or https://. "
            f"Invalid: {', '.join(invalid_urls[:3])}"
        )
        return [], error_message
    
    # All valid URLs are now accepted (both YouTube and articles)
    # Content extraction happens in process_session
    return valid_urls, None
