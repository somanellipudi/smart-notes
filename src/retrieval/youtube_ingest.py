"""
YouTube-specific ingestion module for Smart Notes.

Handles:
- YouTube URL detection and video ID extraction
- Transcript fetching via youtube-transcript-api
- Language fallback (preferred English, fallback to available)
- Graceful degradation when package not installed
- User-friendly error messages
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Optional dependency - make available for patching in tests
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YouTubeTranscriptApi = None  # Make available for patching
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    logger.warning(
        "youtube-transcript-api not available. "
        "Install with: pip install youtube-transcript-api"
    )


@dataclass
class TranscriptResult:
    """Result of transcript extraction attempt."""
    success: bool
    text: str = ""
    video_id: Optional[str] = None
    language: Optional[str] = None
    fetched_at: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "success": self.success,
            "text": self.text,
            "video_id": self.video_id,
            "language": self.language,
            "fetched_at": self.fetched_at,
            "error": self.error,
            "metadata": self.metadata or {}
        }


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from URL.
    
    Supports formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://youtube.com/v/VIDEO_ID
    
    Args:
        url: YouTube URL string
    
    Returns:
        11-character video ID or None if not found
    """
    if not url or not isinstance(url, str):
        return None
    
    # YouTube video IDs are always exactly 11 characters: [a-zA-Z0-9_-]{11}
    patterns = [
        # watch?v= format
        r'youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        # youtu.be format
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
        # embed format
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        # /v/ format
        r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            logger.debug(f"Extracted video ID {video_id} from {url}")
            return video_id
    
    logger.debug(f"Could not extract video ID from URL: {url}")
    return None


def is_youtube_url(url: str) -> bool:
    """
    Check if URL is a YouTube URL.
    
    Args:
        url: URL to check
    
    Returns:
        True if YouTube URL, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    youtube_patterns = [
        r'youtube\.com',
        r'youtu\.be',
        r'youtube-nocookie\.com'
    ]
    
    for pattern in youtube_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    
    return False


def fetch_transcript_text(
    video_id: str,
    languages: Optional[List[str]] = None,
    include_timestamps: bool = False
) -> TranscriptResult:
    """
    Fetch transcript text for a YouTube video with language fallback.
    
    Args:
        video_id: YouTube video ID (11 characters)
        languages: List of language codes to try, in preference order.
                  Default: ["en", "en-US"]
        include_timestamps: If True, prefix each line with [HH:MM:SS]
    
    Returns:
        TranscriptResult with success/failure status and transcript or error message
    
    Examples:
        >>> result = fetch_transcript_text("dQw4w9WgXcQ")
        >>> if result.success:
        ...     print(result.text)
        ... else:
        ...     print(f"Error: {result.error}")
    """
    if not video_id or not isinstance(video_id, str):
        return TranscriptResult(
            success=False,
            error="Invalid video ID provided"
        )
    
    if languages is None:
        languages = ["en", "en-US"]
    
    # Check if package available
    if not YOUTUBE_TRANSCRIPT_AVAILABLE or YouTubeTranscriptApi is None:
        return TranscriptResult(
            success=False,
            video_id=video_id,
            error="youtube-transcript-api not installed. Install with: pip install youtube-transcript-api",
            metadata={"install_command": "pip install youtube-transcript-api"}
        )
    
    try:
        # List available transcripts
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try preferred languages first
        transcript_list = None
        used_language = None
        
        # Try to find transcript in preferred languages
        try:
            transcript_list = transcripts.find_transcript(languages)
            used_language = transcript_list.language
            logger.info(f"Using {used_language} transcript for video {video_id}")
        except Exception as e:
            logger.debug(f"Preferred languages not found: {e}")
            
            # Try any available English variant
            try:
                available = transcripts.get_available_transcripts()
                for t in available:
                    if t.language.startswith('en'):
                        transcript_list = t
                        used_language = t.language
                        logger.info(f"Using fallback language {used_language} for video {video_id}")
                        break
            except Exception:
                pass
            
            # Last resort: use any available transcript
            if not transcript_list:
                try:
                    available = transcripts.get_available_transcripts()
                    if available:
                        transcript_list = available[0]
                        used_language = transcript_list.language
                        logger.warning(
                            f"No English transcript found. Using {used_language} for video {video_id}"
                        )
                except Exception as e:
                    logger.error(f"No transcripts available: {e}")
        
        if not transcript_list:
            return TranscriptResult(
                success=False,
                video_id=video_id,
                error="Transcript not available. Please paste transcript or use an article URL.",
                metadata={"video_id": video_id}
            )
        
        # Fetch transcript entries
        transcript = transcript_list.fetch()
        
        # Format transcript
        if include_timestamps:
            # Format: [HH:MM:SS] Text
            lines = []
            for entry in transcript:
                start_time = entry.get('start', 0)
                hours = int(start_time // 3600)
                minutes = int((start_time % 3600) // 60)
                seconds = int(start_time % 60)
                timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
                text = entry.get('text', '').strip()
                if text:
                    lines.append(f"{timestamp} {text}")
            
            transcript_text = "\n".join(lines)
        else:
            # Just concatenate text
            transcript_text = " ".join(
                entry.get('text', '').strip()
                for entry in transcript
                if entry.get('text')
            )
        
        # Clean whitespace
        transcript_text = re.sub(r'\s+', ' ', transcript_text).strip()
        
        if not transcript_text:
            return TranscriptResult(
                success=False,
                video_id=video_id,
                language=used_language,
                error="Transcript is empty",
                metadata={"video_id": video_id, "language": used_language}
            )
        
        logger.info(
            f"Successfully fetched transcript: {len(transcript_text)} chars "
            f"from {video_id} (language: {used_language})"
        )
        
        return TranscriptResult(
            success=True,
            text=transcript_text,
            video_id=video_id,
            language=used_language,
            fetched_at=datetime.utcnow().isoformat(),
            metadata={
                "video_id": video_id,
                "language": used_language,
                "char_count": len(transcript_text),
                "include_timestamps": include_timestamps
            }
        )
    
    except Exception as e:
        error_msg = str(e).lower()
        
        # Provide specific error messages
        if "disabled" in error_msg or "403" in error_msg:
            user_message = "Captions disabled on this video. Please paste transcript or use an article URL."
        elif "not available" in error_msg or "video_not_available" in error_msg:
            user_message = "Video not available or has been removed. Please use another video or article."
        elif "private" in error_msg:
            user_message = "Video is private. Please use a public video or article URL."
        else:
            user_message = f"Transcript not available. {str(e)}"
        
        logger.warning(f"Failed to fetch transcript for {video_id}: {user_message}")
        
        return TranscriptResult(
            success=False,
            video_id=video_id,
            error=user_message,
            metadata={"video_id": video_id, "original_error": str(e)}
        )


def get_fallback_message(error: Optional[str] = None) -> str:
    """
    Get user-friendly fallback message for transcript extraction failure.
    
    Args:
        error: Original error message (optional)
    
    Returns:
        User-friendly message with guidance
    """
    if error is None:
        error = ""
    
    error_lower = error.lower()
    
    if "disabled" in error_lower:
        return "Captions disabled on this video. Please paste transcript or use an article URL."
    elif "not available" in error_lower or "not found" in error_lower:
        return "Transcript not available. Please paste transcript or use an article URL."
    elif "private" in error_lower:
        return "Video is private. Please use a public video or article URL."
    elif "not installed" in error_lower:
        return "youtube-transcript-api not installed. Install with: pip install youtube-transcript-api"
    else:
        return "Unable to extract transcript from this video. Please paste transcript or use an article URL."
