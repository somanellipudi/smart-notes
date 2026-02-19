"""
URL ingestion for YouTube transcripts and article text extraction.

This module provides intelligent ingestion of URLs:
- YouTube: Fetch transcript via youtube_ingest module (with fallback handling)
- Articles: Extract main content via trafilatura or readability-lxml
"""

import logging
import re
from typing import Tuple, Dict, Any
from urllib.parse import urlparse, parse_qs

from src.retrieval.youtube_ingest import (
    is_youtube_url,
    extract_video_id,
    fetch_transcript_text,
    get_fallback_message,
)

logger = logging.getLogger(__name__)

# Standard user agent to avoid blocking
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
REQUEST_TIMEOUT = 10


def fetch_url_text(url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch and extract text from a URL.
    
    Args:
        url: URL string (YouTube or article)
    
    Returns:
        Tuple of (extracted_text, metadata_dict)
    
    Metadata includes:
        - url: original URL
        - source_type: "youtube" | "article"
        - title: extracted title (if available)
        - video_id: YouTube video ID (if applicable)
        - language: language code (if YouTube)
        - words: word count
        - extraction_method: "youtube_api" | "trafilatura" | "readability" | "beautifulsoup" | "error"
    """
    metadata: Dict[str, Any] = {
        "url": url,
        "source_type": "unknown",
        "title": None,
        "video_id": None,
        "language": None,
        "words": 0,
        "extraction_method": None,
        "error": None
    }
    
    logger.info(f"Fetching URL: {url}")
    
    try:
        if is_youtube_url(url):
            text, meta = _fetch_youtube_transcript(url)
            metadata.update(meta)
            metadata["source_type"] = "youtube"
        else:
            text, meta = _fetch_article_text(url)
            metadata.update(meta)
            metadata["source_type"] = "article"
        
        if text:
            text = _clean_text(text)
            metadata["words"] = len(text.split())
            logger.info(f"✓ URL extraction succeeded: {metadata['words']} words via {metadata['extraction_method']}")
        else:
            metadata["error"] = "No text extracted"
            logger.warning(f"✗ URL extraction failed: {metadata['error']}")
        
        return text, metadata
    
    except Exception as e:
        logger.error(f"Error fetching URL: {e}")
        metadata["error"] = str(e)
        metadata["extraction_method"] = "error"
        return "", metadata


def _fetch_youtube_transcript(url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch YouTube video transcript using dedicated youtube_ingest module.
    
    Returns:
        Tuple of (transcript_text, metadata_dict)
    """
    metadata: Dict[str, Any] = {
        "title": None,
        "video_id": None,
        "language": None,
        "extraction_method": None
    }
    
    # Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        return "", {**metadata, "extraction_method": "error", "error": "Could not extract video ID from URL"}
    
    metadata["video_id"] = video_id
    
    # Use dedicated youtube_ingest module for transcript fetching
    result = fetch_transcript_text(video_id, languages=['en', 'en-US'])
    
    if result.success:
        metadata["language"] = result.language
        metadata["extraction_method"] = "youtube_api"
        logger.info(f"✓ YouTube transcript fetched: {len(result.text)} chars (language: {result.language})")
        return result.text, metadata
    else:
        # Provide user-friendly error message
        error_msg = result.error or "Failed to fetch transcript"
        logger.warning(f"✗ YouTube transcript fetch failed: {error_msg}")
        return "", {**metadata, "extraction_method": "error", "error": error_msg}


def _fetch_article_text(url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch and extract article text from URL.
    
    Returns:
        Tuple of (article_text, metadata_dict)
    """
    metadata: Dict[str, Any] = {
        "title": None,
        "extraction_method": None
    }
    
    # Fetch HTML
    try:
        import requests
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        html = response.text
    except ImportError:
        logger.warning("requests library not available")
        return "", {**metadata, "extraction_method": "error", "error": "requests not installed"}
    except Exception as e:
        logger.warning(f"Failed to fetch URL: {e}")
        return "", {**metadata, "extraction_method": "error", "error": str(e)}
    
    # Strategy 1: Try trafilatura
    try:
        import trafilatura
        result = trafilatura.extract(html, include_comments=False)
        if result:
            # Extract title
            title = trafilatura.extract_metadata(html).get('title', None)
            metadata["title"] = title
            metadata["extraction_method"] = "trafilatura"
            return result, metadata
    except ImportError:
        logger.debug("trafilatura not available")
    except Exception as e:
        logger.debug(f"trafilatura extraction failed: {e}")
    
    # Strategy 2: Try readability-lxml
    try:
        from readability import Document
        doc = Document(html)
        text = doc.summary()
        
        # Strip HTML tags
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
        except Exception:
            # Fallback: simple regex tag removal
            text = re.sub(r'<[^>]+>', '', text)
        
        if text.strip():
            metadata["title"] = doc.title()
            metadata["extraction_method"] = "readability"
            return text, metadata
    except ImportError:
        logger.debug("readability-lxml not available")
    except Exception as e:
        logger.debug(f"readability extraction failed: {e}")
    
    # Strategy 3: Try Beautiful Soup basic extraction
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.get_text()
        
        if text.strip():
            metadata["extraction_method"] = "beautifulsoup"
            return text, metadata
    except ImportError:
        logger.debug("BeautifulSoup not available")
    except Exception as e:
        logger.debug(f"BeautifulSoup extraction failed: {e}")
    
    # Fallback: return empty
    logger.warning("All article extraction strategies failed")
    return "", {**metadata, "extraction_method": "error", "error": "All extraction strategies failed"}


def _clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""

    from src.preprocessing.text_cleaner import clean_extracted_text

    cleaned, _ = clean_extracted_text(text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()
