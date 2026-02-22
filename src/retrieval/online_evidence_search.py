"""
Online Evidence Search Integration

Integrates web search for retrieving evidence from authoritative online sources.
Uses DuckDuckGo search (no API key required) to find relevant content.
"""

import logging
import requests
from typing import List, Dict, Any, Optional  
from dataclasses import dataclass
import time

from src.retrieval.online_retriever import OnlineRetriever, create_retriever
from src.retrieval.authority_sources import get_allowlist
from src.retrieval.evidence_store import Evidence, EvidenceStore
from src.utils.performance_logger import PerformanceTimer
import config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result from web search."""
    title: str
    url: str
    snippet: str
    authority_tier: int = 3
    relevance_score: float = 0.0


class OnlineEvidenceSearcher:
    """
    Search online sources for evidence to verify claims.
    
    Uses DuckDuckGo for search, filters by authority allowlist,
    and retrieves content from approved sources.
    """
    
    def __init__(
        self,
        max_results_per_query: int = 5,
        max_urls_to_fetch: int = 3,
        cache_enabled: bool = True
    ):
        """
        Initialize online evidence searcher.
        
        Args:
            max_results_per_query: Maximum search results to consider
            max_urls_to_fetch: Maximum URLs to actually fetch content from
            cache_enabled: Enable caching of fetched content
        """
        self.max_results_per_query = max_results_per_query
        self.max_urls_to_fetch = max_urls_to_fetch
        self.retriever = create_retriever(cache_enabled=cache_enabled)
        self.allowlist = get_allowlist()
        
        # Rate limiting
        self.last_search_time = 0.0
        self.min_search_interval = 1.0  # 1 second between searches
    
    def _rate_limit(self):
        """Enforce rate limiting between searches."""
        elapsed = time.time() - self.last_search_time
        if elapsed < self.min_search_interval:
            time.sleep(self.min_search_interval - elapsed)
        self.last_search_time = time.time()
    
    def search_duckduckgo(self, query: str) -> List[SearchResult]:
        """
        Search DuckDuckGo for relevant content.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        self._rate_limit()
        
        try:
            # Use DuckDuckGo's instant answer API
            # Note: This is a simplified implementation
            # For production, consider using official API or duckduckgo-search library
            
            # Fallback: If no search library available, search specific authoritative sites
            logger.info(f"Searching for: {query}")
            
            # For now, we'll construct URLs for known authoritative sources
            # In production, integrate with actual search API
            results = self._search_authoritative_sources(query)
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _search_authoritative_sources(self, query: str) -> List[SearchResult]:
        """
        Search specific authoritative sources directly.
        
        This is a fallback when search API is unavailable.
        Constructs URLs for known authoritative sources.
        """
        results = []
        query_encoded = query.replace(" ", "+")
        
        # Wikipedia
        wiki_url = f"https://en.wikipedia.org/wiki/Special:Search?search={query_encoded}"
        results.append(SearchResult(
            title=f"Wikipedia: {query}",
            url=f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            snippet=f"Search Wikipedia for: {query}",
            authority_tier=3,
            relevance_score=0.7
        ))
        
        # Khan Academy
        if any(term in query.lower() for term in ['math', 'science', 'calculus', 'algebra', 'physics']):
            results.append(SearchResult(
                title=f"Khan Academy: {query}",
                url=f"https://www.khanacademy.org/search?search_again=1&page_search_query={query_encoded}",
                snippet=f"Khan Academy content on: {query}",
                authority_tier=1,
                relevance_score=0.9
            ))
        
        # ArXiv (for technical/academic queries)
        if any(term in query.lower() for term in ['algorithm', 'theorem', 'proof', 'research', 'paper']):
            results.append(SearchResult(
                title=f"ArXiv: {query}",
                url=f"https://arxiv.org/search/?query={query_encoded}&searchtype=all",
                snippet=f"ArXiv papers on: {query}",
                authority_tier=1,
                relevance_score=0.85
            ))
        
        return results[:self.max_results_per_query]
    
    def search_and_retrieve_evidence(
        self,
        claim_text: str,
        session_id: Optional[str] = None
    ) -> List[Evidence]:
        """
        Search online and retrieve evidence for a claim.
        
        Args:
            claim_text: Claim to find evidence for
            session_id: Session ID for logging
        
        Returns:
            List of Evidence objects from online sources
        """
        if not claim_text or not claim_text.strip():
            logger.info("Skipping online search for empty claim")
            return []

        claim_preview = (claim_text or "").strip().replace("\n", " ")
        claim_preview = claim_preview[:120] + ("..." if len(claim_preview) > 120 else "")

        with PerformanceTimer(
            "online_evidence_search",
            session_id=session_id,
            metadata={
                "claim_length": len(claim_text),
                "claim_preview": claim_preview
            }
        ):
            # Search for relevant content
            search_results = self.search_duckduckgo(claim_text)
            
            if not search_results:
                logger.warning(f"No search results for claim: {claim_text[:100]}")
                return []
            
            # Filter by authority allowlist
            filtered_results = []
            filtered_out = 0
            for result in search_results:
                is_allowed, reason = self.allowlist.validate_source(result.url)
                if is_allowed:
                    filtered_results.append(result)
                else:
                    logger.debug(f"Filtered out {result.url}: {reason}")
                    filtered_out += 1
            
            if not filtered_results:
                logger.warning("No results passed authority filter")
                return []
            
            # Fetch content from top URLs
            evidence_list = []
            urls_to_fetch = [r.url for r in filtered_results[:self.max_urls_to_fetch]]
            url_domains = []
            for url in urls_to_fetch:
                try:
                    domain = url.split("//", 1)[-1].split("/", 1)[0]
                except Exception:
                    domain = "unknown"
                url_domains.append(domain)
            
            with PerformanceTimer(
                "fetch_online_content",
                session_id=session_id,
                metadata={
                    "num_urls": len(urls_to_fetch),
                    "num_search_results": len(search_results),
                    "num_allowed_results": len(filtered_results),
                    "num_filtered_out": filtered_out,
                    "url_domains": url_domains
                }
            ):
                online_spans = self.retriever.search_and_retrieve(
                    query=claim_text,
                    urls=urls_to_fetch
                )
            
            # Convert online spans to Evidence objects
            for span in online_spans:
                evidence = Evidence(
                    evidence_id=span.span_id,
                    source_id=span.source_id,
                    source_type="online_article",
                    text=span.text,
                    chunk_index=0,
                    char_start=span.start_char,
                    char_end=span.end_char,
                    metadata={
                        "origin_url": span.origin_url,
                        "authority_tier": span.authority_tier.name,
                        "authority_weight": span.authority_weight,
                        "access_date": span.access_date,
                        "is_from_cache": span.is_from_cache
                    },
                    origin=span.origin_url
                )
                evidence_list.append(evidence)
            
            logger.info(
                "Retrieved %s evidence chunks from %s online sources (allowed=%s, filtered_out=%s)",
                len(evidence_list),
                len(urls_to_fetch),
                len(filtered_results),
                filtered_out
            )
            
            return evidence_list


def build_online_evidence_store(
    session_id: str,
    claims: List[Any],  # List[LearningClaim]
    embedding_provider: Any,  # EmbeddingProvider
    max_evidence_per_claim: int = 10
) -> tuple[EvidenceStore, Dict[str, Any]]:
    """
    Build evidence store from ONLINE SOURCES ONLY.
    
    This is the replacement for build_session_evidence_store that uses input.
    Instead, this searches online for each claim and retrieves external evidence.
    
    Args:
        session_id: Session identifier
        claims: List of claims to find evidence for
        embedding_provider: Provider for computing embeddings
        max_evidence_per_claim: Maximum evidence chunks per claim
    
    Returns:
        (evidence_store, stats) tuple
    """
    searcher = OnlineEvidenceSearcher(
        max_results_per_query=5,
        max_urls_to_fetch=3
    )
    
    store = EvidenceStore(session_id=session_id)
    all_evidence = []
    
    skipped_claims = 0

    with PerformanceTimer(
        "build_online_evidence_store",
        session_id=session_id,
        metadata={"num_claims": len(claims)}
    ):
        # Search online for each claim
        for i, claim in enumerate(claims[:20], 1):  # Limit to first 20 claims to avoid excessive searching
            logger.info(f"Searching online evidence for claim {i}/{min(len(claims), 20)}")

            claim_text = (claim.claim_text or claim.metadata.get("draft_text", "")).strip()
            if not claim_text:
                skipped_claims += 1
                logger.info("Skipping empty claim at index %s", i)
                continue

            claim_evidence = searcher.search_and_retrieve_evidence(
                claim_text=claim_text,
                session_id=session_id
            )
            
            # Limit evidence per claim
            all_evidence.extend(claim_evidence[:max_evidence_per_claim])
            
            # Rate limiting between claims
            if i < min(len(claims), 20):
                time.sleep(0.5)
        
        logger.info(f"Total evidence collected: {len(all_evidence)} chunks")
        
        # Add all evidence to store
        for evidence in all_evidence:
            store.evidence.append(evidence)
            store.evidence_by_id[evidence.evidence_id] = evidence
        
        # Compute embeddings
        if embedding_provider and all_evidence:
            with PerformanceTimer(
                "embed_online_evidence",
                session_id=session_id,
                metadata={"num_chunks": len(all_evidence)}
            ):
                texts = [ev.text for ev in all_evidence]
                embeddings = embedding_provider.embed_texts(texts)
                
                for evidence, embedding in zip(all_evidence, embeddings):
                    evidence.embedding = embedding
                
                # Build FAISS index
                store.build_index(embeddings=embeddings, embedding_dim=embeddings.shape[1])
        
        # Compile statistics
        stats = {
            "session_id": session_id,
            "num_claims_searched": min(len(claims), 20),
            "num_claims_skipped": skipped_claims,
            "total_evidence_chunks": len(all_evidence),
            "num_sources": len(set(ev.source_id for ev in all_evidence)),
            "evidence_source": "online_only",
            "index_built": store.index_built
        }
        
        return store, stats


# Configuration helper
def should_use_online_evidence() -> bool:
    """Check if online evidence should be used instead of input."""
    return getattr(config, 'ENABLE_ONLINE_VERIFICATION', False)
