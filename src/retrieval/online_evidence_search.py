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
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        max_results_per_query: int = 10,
        max_urls_to_fetch: int = 5,
        cache_enabled: bool = True
    ):
        """
        Initialize online evidence searcher.
        
        Args:
            max_results_per_query: Maximum search results to consider (default: 10 to include multiple sources)
            max_urls_to_fetch: Maximum URLs to actually fetch content from (default: 5 for diverse sources)
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
        Search specific authoritative sources using constructive URLs.
        
        Instead of search pages, we construct direct resource URLs and
        use a fallback approach if direct URLs don't exist.
        """
        results = []
        query_lower = query.lower()
        
        # Known authoritative sources with fetchable content patterns
        authoritative_sources = [
            # Official Python documentation - direct searchable endpoint
            {
                "domain": "docs.python.org",
                "title": "Python Official Docs",
                "url_pattern": lambda q: f"https://docs.python.org/3/search.html?q={q.replace(' ', '+')}",
                "tier": 1,
                "score": 0.95,
                "always_include": True
            },
            # MDN Web Docs - direct searchable endpoint
            {
                "domain": "developer.mozilla.org",
                "title": "MDN Web Docs",
                "url_pattern": lambda q: f"https://developer.mozilla.org/en-US/search?q={q.replace(' ', '%20')}",
                "tier": 1,
                "score": 0.92,
                "always_include": True
            },
            # GitHub official docs - direct searchable endpoint
            {
                "domain": "docs.github.com",
                "title": "GitHub Docs",
                "url_pattern": lambda q: f"https://docs.github.com/search?query={q.replace(' ', '+')}",
                "tier": 1,
                "score": 0.90,
                "always_include": True
            },
            # ArXiv - direct search API
            {
                "domain": "arxiv.org",
                "title": "ArXiv Research Papers",
                "url_pattern": lambda q: f"https://arxiv.org/search/advanced?terms-0-operator=AND&terms-0-term={q.replace(' ', '+')}&classification-include-cross-list=include",
                "tier": 2,
                "score": 0.88,
                "always_include": True
            },
            # Khan Academy - direct searchable endpoint
            {
                "domain": "khanacademy.org",
                "title": "Khan Academy",
                "url_pattern": lambda q: f"https://www.khanacademy.org/search?page_search_query={q.replace(' ', '+')}",
                "tier": 2,
                "score": 0.85,
                "always_include": True
            },
            # Coursera - direct searchable endpoint
            {
                "domain": "coursera.org",
                "title": "Coursera Courses",
                "url_pattern": lambda q: f"https://www.coursera.org/search?query={q.replace(' ', '+')}",
                "tier": 2,
                "score": 0.82,
                "always_include": True
            },
            # Stack Overflow - direct searchable endpoint
            {
                "domain": "stackoverflow.com",
                "title": "Stack Overflow Q&A",
                "url_pattern": lambda q: f"https://stackoverflow.com/search?q={q.replace(' ', '+')}",
                "tier": 3,
                "score": 0.80,
                "always_include": True
            },
            # GeeksforGeeks - direct searchable endpoint
            {
                "domain": "geeksforgeeks.org",
                "title": "GeeksforGeeks Tutorials",
                "url_pattern": lambda q: f"https://www.geeksforgeeks.org/search/?q={q.replace(' ', '+')}",
                "tier": 3,
                "score": 0.78,
                "always_include": True
            },
            # Wikipedia - direct article endpoint (not search page)
            {
                "domain": "en.wikipedia.org",
                "title": "Wikipedia",
                "url_pattern": lambda q: f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={q.replace(' ', '_')}&prop=extracts&explaintext=true",
                "tier": 3,
                "score": 0.75,
                "always_include": True
            },
            # MIT OpenCourseWare - direct searchable endpoint
            {
                "domain": "ocw.mit.edu",
                "title": "MIT OpenCourseWare",
                "url_pattern": lambda q: f"https://ocw.mit.edu/search/?s={q.replace(' ', '+')}",
                "tier": 2,
                "score": 0.90,
                "always_include": True
            },
            # LeetCode (conditional - for algorithm queries)
            {
                "domain": "leetcode.com",
                "title": "LeetCode Problems",
                "url_pattern": lambda q: f"https://leetcode.com/search/?q={q.replace(' ', '+')}",
                "tier": 3,
                "score": 0.76,
                "always_include": False,
                "keywords": ['stack', 'queue', 'algorithm', 'data structure', 'tree', 'graph', 'sort', 'search']
            },
            # Wolfram MathWorld (conditional - for math queries)
            {
                "domain": "mathworld.wolfram.com",
                "title": "Wolfram MathWorld",
                "url_pattern": lambda q: f"https://mathworld.wolfram.com/search/?query={q.replace(' ', '+')}",
                "tier": 2,
                "score": 0.84,
                "always_include": False,
                "keywords": ['math', 'calculus', 'algebra', 'geometry', 'equation', 'formula', 'theorem', 'derivative', 'integral']
            },
            # edX (conditional - for education queries)
            {
                "domain": "edx.org",
                "title": "edX Courses",
                "url_pattern": lambda q: f"https://www.edx.org/search?q={q.replace(' ', '+')}",
                "tier": 2,
                "score": 0.80,
                "always_include": False,
                "keywords": ['course', 'learn', 'education', 'training', 'study']
            },
        ]
        
        # Add always-included sources
        for source in authoritative_sources:
            if source.get("always_include", False):
                try:
                    url = source["url_pattern"](query)
                    results.append(SearchResult(
                        title=source["title"],
                        url=url,
                        snippet=f"Search {source['title']}: {query}",
                        authority_tier=source["tier"],
                        relevance_score=source["score"]
                    ))
                    logger.debug(f"Added {source['title']} source for query: {query[:40]}...")
                except Exception as e:
                    logger.warning(f"Failed to generate URL for {source['title']}: {e}")
        
        # Add conditional sources based on query keywords
        for source in authoritative_sources:
            if not source.get("always_include", False):
                keywords = source.get("keywords", [])
                if any(kw in query_lower for kw in keywords):
                    try:
                        url = source["url_pattern"](query)
                        results.append(SearchResult(
                            title=source["title"],
                            url=url,
                            snippet=f"Search {source['title']}: {query}",
                            authority_tier=source["tier"],
                            relevance_score=source["score"]
                        ))
                        logger.debug(f"Added conditional {source['title']} source for query: {query[:40]}...")
                    except Exception as e:
                        logger.warning(f"Failed to generate URL for {source['title']}: {e}")
        
        logger.info(f"Constructed {len(results)} authoritative source URLs for query: '{query[:50]}...'")
        source_names = [r.title for r in results]
        logger.info(f"Sources included: {', '.join(source_names[:10])}")
        
        return results[:self.max_results_per_query]
    
    def search_and_retrieve_evidence(
        self,
        claim_text: str,
        session_id: Optional[str] = None,
        use_query_expansion: bool = True
    ) -> List[Evidence]:
        """
        Search online and retrieve evidence for a claim.
        
        Args:
            claim_text: Claim to find evidence for
            session_id: Session ID for logging
            use_query_expansion: Enable ML-based query expansion
        
        Returns:
            List of Evidence objects from online sources
        """
        if not claim_text or not claim_text.strip():
            logger.info("Skipping online search for empty claim")
            return []

        # ML Optimization: Expand query for better search results
        search_query = claim_text
        if use_query_expansion:
            try:
                from src.utils.ml_advanced_optimizations import get_query_expander
                expander = get_query_expander()
                search_query = expander.expand_query(claim_text)
                logger.debug(f"Query expanded: '{claim_text[:50]}...' → '{search_query[:80]}...'")
            except Exception as e:
                logger.debug(f"Query expansion failed, using original: {e}")
                search_query = claim_text

        claim_preview = (claim_text or "").strip().replace("\n", " ")
        claim_preview = claim_preview[:120] + ("..." if len(claim_preview) > 120 else "")

        with PerformanceTimer(
            "online_evidence_search",
            session_id=session_id,
            metadata={
                "claim_length": len(claim_text),
                "claim_preview": claim_preview,
                "query_expanded": use_query_expansion
            }
        ):
            # Search for relevant content (using expanded query)
            search_results = self.search_duckduckgo(search_query)
            
            if not search_results:
                logger.warning(f"No search results for claim: {claim_text[:100]}")
                return []
            
            logger.info(f"🔍 Searched {len(search_results)} sources for claim: '{claim_text[:80]}...'")
            logger.info(f"   Sources searched: {', '.join([r.title for r in search_results])}")
            
            # Filter by authority allowlist
            filtered_results = []
            filtered_out = 0
            for result in search_results:
                is_allowed, reason = self.allowlist.validate_source(result.url)
                if is_allowed:
                    filtered_results.append(result)
                    logger.debug(f"✅ Allowed: {result.title}")
                else:
                    logger.debug(f"❌ Filtered out {result.title}: {reason}")
                    filtered_out += 1
            
            if not filtered_results:
                logger.warning("No results passed authority filter")
                return []
            
            logger.info(f"✓ {len(filtered_results)} sources passed authority filter: {', '.join([r.title for r in filtered_results])}")
            
            # Fetch content from top URLs
            evidence_list = []
            urls_to_fetch = [r.url for r in filtered_results[:self.max_urls_to_fetch]]
            url_domains = []
            source_titles = []
            for i, url in enumerate(urls_to_fetch):
                try:
                    domain = url.split("//", 1)[-1].split("/", 1)[0]
                    if i < len(filtered_results):
                        source_titles.append(filtered_results[i].title)
                except Exception:
                    domain = "unknown"
                url_domains.append(domain)
            
            logger.info(f"📡 Fetching from {len(urls_to_fetch)} sources: {', '.join(source_titles)}")
            
            with PerformanceTimer(
                "fetch_online_content",
                session_id=session_id,
                metadata={
                    "num_urls": len(urls_to_fetch),
                    "num_search_results": len(search_results),
                    "num_allowed_results": len(filtered_results),
                    "num_filtered_out": filtered_out,
                    "url_domains": url_domains,
                    "source_titles": source_titles
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
            
            # Extract unique sources from evidence
            unique_sources = set()
            for ev in evidence_list:
                source_url = ev.origin
                try:
                    domain = source_url.split("//", 1)[-1].split("/", 1)[0]
                    unique_sources.add(domain)
                except:
                    pass
            
            logger.info(
                "✅ Retrieved %s evidence chunks from %s unique online sources: %s",
                len(evidence_list),
                len(unique_sources),
                ', '.join(sorted(unique_sources)) if unique_sources else "none"
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
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    searcher = OnlineEvidenceSearcher(
        max_results_per_query=5,
        max_urls_to_fetch=3
    )
    
    store = EvidenceStore(session_id=session_id)
    all_evidence = []
    
    skipped_claims = 0
    max_claims_to_search = min(len(claims), config.ONLINE_MAX_CLAIMS_TO_SEARCH)

    with PerformanceTimer(
        "build_online_evidence_store",
        session_id=session_id,
        metadata={"num_claims": len(claims)}
    ):
        # PARALLEL SEARCH: Use ThreadPoolExecutor for I/O-bound search operations
        claims_to_search = claims[:max_claims_to_search]
        
        def search_claim(i_claim_tuple):
            """Search for a single claim (for parallel execution)."""
            i, claim = i_claim_tuple
            claim_text = (claim.claim_text or claim.metadata.get("draft_text", "")).strip()
            if not claim_text:
                logger.info(f"Skipping empty claim at index {i+1}")
                return None, i+1
            
            logger.info(f"Searching online evidence for claim {i+1}/{max_claims_to_search}")
            claim_evidence = searcher.search_and_retrieve_evidence(
                claim_text=claim_text,
                session_id=session_id
            )
            return claim_evidence[:max_evidence_per_claim], i+1
        
        # Execute searches in parallel (max 4 concurrent threads to respect rate limits)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(search_claim, (i, claim)): i 
                for i, claim in enumerate(claims_to_search)
            }
            
            for future in as_completed(futures):
                try:
                    claim_evidence, claim_idx = future.result()
                    if claim_evidence is None:
                        skipped_claims += 1
                    else:
                        all_evidence.extend(claim_evidence)
                        logger.info(f"Claim {claim_idx}: Retrieved {len(claim_evidence)} evidence chunks")
                except Exception as e:
                    logger.error(f"Error searching claim: {e}", exc_info=True)
                    skipped_claims += 1
        
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
            "num_claims_searched": max_claims_to_search,
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


# Module-level convenience function for cited generation pipeline
def search_and_retrieve_evidence(
    query: str,
    session_id: Optional[str] = None,
    max_results: int = 3,
    use_query_expansion: bool = True
) -> List[Dict[str, Any]]:
    """
    Module-level convenience function for searching and retrieving online evidence.
    
    This is a simpler interface for use in cited generation pipeline.
    
    Args:
        query: Search query (topic or concept)
        session_id: Session ID for logging
        max_results: Maximum results to return
        use_query_expansion: Enable ML-based query expansion
    
    Returns:
        List of evidence dicts with keys: snippet, url, source_id, text
    """
    searcher = OnlineEvidenceSearcher(
        max_results_per_query=5,
        max_urls_to_fetch=max_results
    )
    
    evidence_list = searcher.search_and_retrieve_evidence(
        claim_text=query,
        session_id=session_id,
        use_query_expansion=use_query_expansion
    )
    
    # Convert Evidence objects to simple dicts
    results = []
    for ev in evidence_list:
        results.append({
            "snippet": ev.text,
            "text": ev.text,
            "url": ev.metadata.get("origin_url", ""),
            "title": ev.metadata.get("source_title", ""),
            "authority_tier": ev.metadata.get("authority_tier", "Unknown"),
            "authority_weight": float(ev.metadata.get("authority_weight", 0.5)),
            "source_id": ev.source_id,
            "evidence_id": ev.evidence_id
        })
    
    return results
