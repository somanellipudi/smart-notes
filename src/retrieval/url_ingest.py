"""
URL ingestion module for Smart Notes.

Supports fetching content from:
- YouTube videos (via transcripts)
- Web articles (via HTML parsing)
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, List, Optional, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available, URL ingestion will be disabled")

try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    logger.warning("readability-lxml not available, falling back to basic HTML parsing")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 not available, article parsing will be limited")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    logger.warning("youtube-transcript-api not available, YouTube ingestion will be limited")


# Configuration
MAX_DOWNLOAD_SIZE_MB = 2
MAX_DOWNLOAD_SIZE_BYTES = MAX_DOWNLOAD_SIZE_MB * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 10
MIN_TEXT_LENGTH_CHARS = 200


@dataclass
class UrlSource:
    """Represents content fetched from a URL."""
    url: str
    source_type: Literal["youtube", "article", "unknown"]
    title: Optional[str]
    fetched_at: str  # ISO format datetime
    text: str
    error: Optional[str] = None  # If fetching failed


def detect_source_type(url: str) -> Literal["youtube", "article", "unknown"]:
    """
    Detect source type from URL.
    
    Args:
        url: URL to analyze
    
    Returns:
        Source type: "youtube", "article", or "unknown"
    """
    try:
        parsed = urlparse(url.lower())
        hostname = parsed.hostname or ""
        
        if "youtube.com" in hostname or "youtu.be" in hostname:
            return "youtube"
        elif parsed.scheme in ("http", "https"):
            return "article"
        else:
            return "unknown"
    except Exception as e:
        logger.warning(f"Failed to parse URL {url}: {e}")
        return "unknown"


def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from URL.
    
    Supports formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    
    Args:
        url: YouTube URL
    
    Returns:
        Video ID or None if not found
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def fetch_youtube_transcript(url: str) -> UrlSource:
    """
    Fetch YouTube video transcript.
    
    Args:
        url: YouTube video URL
    
    Returns:
        UrlSource with transcript text
    """
    video_id = extract_youtube_video_id(url)
    
    if not video_id:
        return UrlSource(
            url=url,
            source_type="youtube",
            title=None,
            fetched_at=datetime.utcnow().isoformat(),
            text="",
            error="Could not extract video ID from URL"
        )
    
    if not YOUTUBE_TRANSCRIPT_AVAILABLE:
        return UrlSource(
            url=url,
            source_type="youtube",
            title=None,
            fetched_at=datetime.utcnow().isoformat(),
            text="",
            error="youtube-transcript-api not installed. Install with: pip install youtube-transcript-api"
        )
    
    try:
        # Fetch transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Concatenate all text entries
        text = " ".join(entry['text'] for entry in transcript_list)
        
        # Basic cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        logger.info(f"Fetched YouTube transcript: {len(text)} chars from video {video_id}")
        
        return UrlSource(
            url=url,
            source_type="youtube",
            title=f"YouTube Video {video_id}",
            fetched_at=datetime.utcnow().isoformat(),
            text=text
        )
    
    except Exception as e:
        logger.warning(f"Failed to fetch YouTube transcript for {video_id}: {e}")
        return UrlSource(
            url=url,
            source_type="youtube",
            title=None,
            fetched_at=datetime.utcnow().isoformat(),
            text="",
            error=f"Failed to fetch transcript: {str(e)}"
        )


def fetch_article_content(url: str) -> UrlSource:
    """
    Fetch web article content using readability + BeautifulSoup.
    
    Args:
        url: Article URL
    
    Returns:
        UrlSource with article text
    """
    if not REQUESTS_AVAILABLE:
        return UrlSource(
            url=url,
            source_type="article",
            title=None,
            fetched_at=datetime.utcnow().isoformat(),
            text="",
            error="requests library not installed. Install with: pip install requests"
        )
    
    try:
        # Fetch HTML with size limit and timeout
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            stream=True
        )
        response.raise_for_status()
        
        # Check content length
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE_BYTES:
            return UrlSource(
                url=url,
                source_type="article",
                title=None,
                fetched_at=datetime.utcnow().isoformat(),
                text="",
                error=f"Content too large: {int(content_length) / 1024 / 1024:.1f}MB (max: {MAX_DOWNLOAD_SIZE_MB}MB)"
            )
        
        # Download content with size limit
        content = b""
        for chunk in response.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > MAX_DOWNLOAD_SIZE_BYTES:
                return UrlSource(
                    url=url,
                    source_type="article",
                    title=None,
                    fetched_at=datetime.utcnow().isoformat(),
                    text="",
                    error=f"Content exceeded max download size: {MAX_DOWNLOAD_SIZE_MB}MB"
                )
        
        html = content.decode('utf-8', errors='ignore')
        
        # Extract main content using readability if available
        title = None
        text = ""
        
        if READABILITY_AVAILABLE:
            try:
                doc = Document(html)
                title = doc.short_title()
                text = doc.summary()
                
                # Parse HTML to extract text
                if BS4_AVAILABLE:
                    soup = BeautifulSoup(text, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                else:
                    # Fallback: basic HTML tag removal
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
            except Exception as e:
                logger.warning(f"Readability parsing failed for {url}: {e}, falling back to BeautifulSoup")
        
        # Fallback to BeautifulSoup if readability failed or unavailable
        if not text and BS4_AVAILABLE:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title if not already set
            if not title and soup.title:
                title = soup.title.string
            
            # Get text from body
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Validate minimum length
        if len(text) < MIN_TEXT_LENGTH_CHARS:
            logger.warning(f"Article text too short ({len(text)} chars) from {url}")
        
        logger.info(f"Fetched article: {len(text)} chars from {url}")
        
        return UrlSource(
            url=url,
            source_type="article",
            title=title or url,
            fetched_at=datetime.utcnow().isoformat(),
            text=text
        )
    
    except requests.exceptions.Timeout:
        return UrlSource(
            url=url,
            source_type="article",
            title=None,
            fetched_at=datetime.utcnow().isoformat(),
            text="",
            error=f"Request timeout after {REQUEST_TIMEOUT_SECONDS}s"
        )
    except requests.exceptions.RequestException as e:
        return UrlSource(
            url=url,
            source_type="article",
            title=None,
            fetched_at=datetime.utcnow().isoformat(),
            text="",
            error=f"Request failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching article from {url}: {e}")
        return UrlSource(
            url=url,
            source_type="article",
            title=None,
            fetched_at=datetime.utcnow().isoformat(),
            text="",
            error=f"Unexpected error: {str(e)}"
        )


def ingest_urls(urls: List[str]) -> List[UrlSource]:
    """
    Ingest content from multiple URLs.
    
    Supports:
    - YouTube videos (fetches transcripts)
    - Web articles (extracts main content)
    
    Args:
        urls: List of URLs to ingest
    
    Returns:
        List of UrlSource objects (may include errors)
    """
    if not urls:
        return []
    
    results = []
    
    for url in urls:
        if not url or not isinstance(url, str):
            logger.warning(f"Skipping invalid URL: {url}")
            continue
        
        url = url.strip()
        
        # Validate URL scheme
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                logger.warning(f"Skipping URL with invalid scheme: {url}")
                results.append(UrlSource(
                    url=url,
                    source_type="unknown",
                    title=None,
                    fetched_at=datetime.utcnow().isoformat(),
                    text="",
                    error="Invalid URL scheme (must be http or https)"
                ))
                continue
        except Exception as e:
            logger.warning(f"Skipping malformed URL: {url} ({e})")
            results.append(UrlSource(
                url=url,
                source_type="unknown",
                title=None,
                fetched_at=datetime.utcnow().isoformat(),
                text="",
                error=f"Malformed URL: {str(e)}"
            ))
            continue
        
        # Detect source type and fetch
        source_type = detect_source_type(url)
        
        logger.info(f"Ingesting {source_type} URL: {url}")
        
        if source_type == "youtube":
            result = fetch_youtube_transcript(url)
        elif source_type == "article":
            result = fetch_article_content(url)
        else:
            result = UrlSource(
                url=url,
                source_type="unknown",
                title=None,
                fetched_at=datetime.utcnow().isoformat(),
                text="",
                error="Unknown source type"
            )
        
        results.append(result)
        
        # Log success or failure
        if result.error:
            logger.warning(f"Failed to ingest {url}: {result.error}")
        elif len(result.text) < MIN_TEXT_LENGTH_CHARS:
            logger.warning(f"Ingested {url} but text is short ({len(result.text)} chars)")
        else:
            logger.info(f"Successfully ingested {url}: {len(result.text)} chars")
    
    return results


def chunk_url_sources(
    url_sources: List[UrlSource],
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Convert URL sources to text chunks for evidence retrieval.
    
    Args:
        url_sources: List of UrlSource objects
        chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks
    
    Returns:
        List of text chunks with source metadata embedded
    """
    chunks = []
    
    for source in url_sources:
        # Skip sources with errors or empty text
        if source.error or not source.text:
            logger.warning(f"Skipping {source.url}: {source.error or 'empty text'}")
            continue
        
        # Skip very short text
        if len(source.text) < MIN_TEXT_LENGTH_CHARS:
            logger.warning(f"Skipping {source.url}: text too short ({len(source.text)} chars)")
            continue
        
        # Create metadata header for chunks
        header = f"[Source: {source.source_type} - {source.title or source.url}]\n"
        
        # Simple chunking with overlap
        text = source.text
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for last period/question/exclamation mark
                for sep in ['. ', '? ', '! ', '\n']:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > chunk_size * 0.7:  # At least 70% of chunk
                        chunk_text = chunk_text[:last_sep + 1]
                        end = start + last_sep + 1
                        break
            
            # Add header and chunk
            full_chunk = header + chunk_text.strip()
            chunks.append(full_chunk)
            
            chunk_index += 1
            start = end - overlap  # Overlap with next chunk
        
        logger.info(f"Chunked {source.url} into {chunk_index} chunks")
    
    return chunks


def get_url_ingestion_summary(url_sources: List[UrlSource]) -> Dict[str, Any]:
    """
    Get summary statistics for URL ingestion.
    
    Args:
        url_sources: List of UrlSource objects
    
    Returns:
        Dictionary with summary statistics
    """
    total = len(url_sources)
    successful = sum(1 for s in url_sources if not s.error and s.text)
    failed = sum(1 for s in url_sources if s.error or not s.text)
    
    by_type = {}
    for source in url_sources:
        by_type[source.source_type] = by_type.get(source.source_type, 0) + 1
    
    total_chars = sum(len(s.text) for s in url_sources if s.text)
    
    return {
        "total_urls": total,
        "successful": successful,
        "failed": failed,
        "by_type": by_type,
        "total_chars": total_chars,
        "avg_chars": total_chars // successful if successful > 0 else 0,
        "urls": [
            {
                "url": s.url,
                "type": s.source_type,
                "title": s.title,
                "chars": len(s.text),
                "success": not s.error and bool(s.text),
                "error": s.error
            }
            for s in url_sources
        ]
    }

