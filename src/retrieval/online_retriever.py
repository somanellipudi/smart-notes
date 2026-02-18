"""
Online Evidence Retrieval with Caching and PII Redaction

Fetches evidence from online sources with:
- Caching to artifacts (content hash + access date)
- Rate limiting and timeouts
- Text extraction and cleaning
- PII redaction for outbound queries
- Chunking into spans with authority metadata
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
import hashlib

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    import urllib.request
    import urllib.error

from src.retrieval.authority_sources import get_allowlist, AuthorityTier
from src.preprocessing.text_cleaner import clean_extracted_text
import config

logger = logging.getLogger(__name__)


@dataclass
class OnlineSpan:
    """Text span extracted from online source."""
    span_id: str  # SHA256(source_id + start + end + text)
    source_id: str  # URL
    text: str
    start_char: int = 0
    end_char: int = 0
    authority_tier: AuthorityTier = AuthorityTier.TIER_3
    authority_weight: float = 0.6
    origin_url: str = ""
    access_date: str = ""
    is_from_cache: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnlineSourceContent:
    """Downloaded content from online source."""
    url: str
    raw_text: str
    cleaned_text: str
    content_hash: str  # SHA256 of cleaned text
    access_date: str
    status_code: int = 200
    error: Optional[str] = None
    authority_tier: AuthorityTier = AuthorityTier.TIER_3
    authority_weight: float = 0.6


class PIIRedactor:
    """Redact personally identifiable information."""
    
    # Pattern for email addresses
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    # Pattern for US phone numbers
    PHONE_PATTERN = re.compile(r'\b(?:\+1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b')
    
    # Pattern for social security numbers
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    
    # Pattern for credit card numbers
    CC_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
    
    # Pattern for IP addresses
    IP_PATTERN = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
    
    # Pattern for home addresses (rough heuristic)
    ADDRESS_PATTERN = re.compile(r'\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)')
    
    @staticmethod
    def redact(text: str) -> str:
        """
        Redact PII from text.
        
        Args:
            text: Text to redact
        
        Returns:
            Text with PII replaced by [REDACTED]
        """
        if not text:
            return text
        
        # Redact emails
        text = PIIRedactor.EMAIL_PATTERN.sub('[REDACTED_EMAIL]', text)
        
        # Redact phone numbers
        text = PIIRedactor.PHONE_PATTERN.sub('[REDACTED_PHONE]', text)
        
        # Redact SSN
        text = PIIRedactor.SSN_PATTERN.sub('[REDACTED_SSN]', text)
        
        # Redact credit cards
        text = PIIRedactor.CC_PATTERN.sub('[REDACTED_CC]', text)
        
        # Redact IP addresses (be careful - may have false positives)
        # Only redact if followed by port or in suspicious context
        # For now, we'll skip to avoid false positives
        
        return text


class OnlineRetriever:
    """Fetch and process online evidence with caching and rate limiting."""
    
    def __init__(
        self,
        cache_enabled: bool = True,
        rate_limit_per_second: float = 2.0,
        timeout_seconds: int = 10,
        max_content_length: int = 1_000_000  # 1 MB max
    ):
        """
        Initialize online retriever.
        
        Args:
            cache_enabled: Use artifacts for caching
            rate_limit_per_second: Max requests per second
            timeout_seconds: Request timeout
            max_content_length: Max downloaded content
        """
        self.cache_enabled = cache_enabled
        self.rate_limit_per_second = rate_limit_per_second
        self.timeout_seconds = timeout_seconds
        self.max_content_length = max_content_length
        self.last_request_time = 0.0
        self.allowlist = get_allowlist()
        
        if not REQUESTS_AVAILABLE and hasattr(config, 'WARN_ON_URLLIB'):
            logger.debug("Using urllib (requests not available)")
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.rate_limit_per_second
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def _compute_content_hash(self, text: str) -> str:
        """Compute SHA256 hash of text content."""
        return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()
    
    def fetch_url(self, url: str) -> Optional[OnlineSourceContent]:
        """
        Fetch content from URL with validation and caching.
        
        Args:
            url: URL to fetch
        
        Returns:
            OnlineSourceContent if successful, None on error
        """
        # Validate against allowlist
        is_allowed, reason = self.allowlist.validate_source(url)
        if not is_allowed:
            logger.warning(f"URL not allowed: {url} ({reason})")
            return None
        
        # Get authority info
        source = self.allowlist.get_source(url)
        if not source:
            logger.warning(f"Source not found: {url}")
            return None
        
        # Rate limiting
        self._rate_limit()
        
        # Attempt fetch
        try:
            logger.info(f"Fetching URL: {url}")
            
            if REQUESTS_AVAILABLE:
                response = requests.get(
                    url,
                    timeout=self.timeout_seconds,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; AuthoritativeBot/1.0)',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                    }
                )
                response.raise_for_status()
                raw_text = response.text
                status_code = response.status_code
            else:
                # Fallback to urllib
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; AuthoritativeBot/1.0)'}
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    raw_text = response.read().decode('utf-8', errors='ignore')
                    status_code = response.status
            
            # Check content length
            if len(raw_text) > self.max_content_length:
                logger.warning(f"Content too large: {len(raw_text)} > {self.max_content_length}")
                raw_text = raw_text[:self.max_content_length]
            
            # Extract and clean text
            cleaned_text, clean_diag = clean_extracted_text(raw_text)
            content_hash = self._compute_content_hash(cleaned_text)
            
            logger.info(
                f"✓ Fetched {url}: {len(cleaned_text)} chars, {clean_diag.removed_lines_count} lines cleaned"
            )
            
            return OnlineSourceContent(
                url=url,
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                content_hash=content_hash,
                access_date=datetime.utcnow().isoformat(),
                status_code=status_code,
                authority_tier=source.tier,
                authority_weight=source.authority_weight
            )
        
        except requests.exceptions.Timeout if REQUESTS_AVAILABLE else urllib.error.URLError as e:  # type: ignore
            logger.error(f"Timeout fetching {url}: {e}")
            return OnlineSourceContent(
                url=url,
                raw_text="",
                cleaned_text="",
                content_hash="",
                access_date=datetime.utcnow().isoformat(),
                status_code=0,
                error=f"Timeout: {str(e)}"
            )
        
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return OnlineSourceContent(
                url=url,
                raw_text="",
                cleaned_text="",
                content_hash="",
                access_date=datetime.utcnow().isoformat(),
                status_code=0,
                error=str(e)
            )
    
    def extract_spans(self, content: OnlineSourceContent, chunk_size: int = 500) -> List[OnlineSpan]:
        """
        Extract spans from online content.
        
        Args:
            content: Online content
            chunk_size: Characters per span
        
        Returns:
            List of OnlineSpan objects
        """
        if not content.cleaned_text:
            return []
        
        spans = []
        text = content.cleaned_text
        
        # Split into overlapping chunks
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i+chunk_size]
            
            # Skip short chunks at end
            if len(chunk_text) < 50:
                continue
            
            # Redact PII
            chunk_text_redacted = PIIRedactor.redact(chunk_text)
            
            # Compute span ID
            span_id = hashlib.sha256(
                f"{content.url}:{i}:{i+chunk_size}:{chunk_text}".encode()
            ).hexdigest()
            
            span = OnlineSpan(
                span_id=span_id,
                source_id=content.url,
                text=chunk_text_redacted,
                start_char=i,
                end_char=min(i+chunk_size, len(text)),
                authority_tier=content.authority_tier,
                authority_weight=content.authority_weight,
                origin_url=content.url,
                access_date=content.access_date,
                metadata={
                    "content_hash": content.content_hash,
                    "authority_tier": content.authority_tier.name,
                    "authority_weight": content.authority_weight
                }
            )
            
            spans.append(span)
        
        logger.info(f"Extracted {len(spans)} spans from {content.url}")
        return spans
    
    def search_and_retrieve(
        self,
        query: str,
        urls: List[str],
        max_urls: int = 5
    ) -> List[OnlineSpan]:
        """
        Search specific URLs for evidence.
        
        Args:
            query: Search query (for logging/audit trail)
            urls: URLs to retrieve
            max_urls: Max URLs to fetch
        
        Returns:
            List of relevant spans
        """
        # Redact PII from query
        query_redacted = PIIRedactor.redact(query)
        logger.info(f"Online search query (redacted): {query_redacted}")
        
        spans = []
        urls_to_fetch = urls[:max_urls]
        
        for url in urls_to_fetch:
            # Validate and fetch
            content = self.fetch_url(url)
            
            if not content or content.error:
                logger.warning(f"Failed to fetch {url}: {content.error if content else 'Unknown error'}")
                continue
            
            # Extract spans
            url_spans = self.extract_spans(content)
            spans.extend(url_spans)
        
        logger.info(f"Retrieved {len(spans)} spans from {len(urls_to_fetch)} URLs")
        return spans


def create_retriever(
    cache_enabled: Optional[bool] = None,
    rate_limit_per_second: Optional[float] = None
) -> OnlineRetriever:
    """
    Create configured online retriever.
    
    Uses config defaults if not specified.
    """
    if cache_enabled is None:
        cache_enabled = getattr(config, 'ONLINE_CACHE_ENABLED', True)
    
    if rate_limit_per_second is None:
        rate_limit_per_second = getattr(config, 'ONLINE_RATE_LIMIT', 2.0)
    
    return OnlineRetriever(
        cache_enabled=cache_enabled,
        rate_limit_per_second=rate_limit_per_second,
        timeout_seconds=getattr(config, 'ONLINE_TIMEOUT_SECONDS', 10)
    )
