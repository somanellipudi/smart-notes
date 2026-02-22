"""
ML-based optimizations for pipeline performance.

Techniques:
1. Semantic deduplication: Skip LLM calls for near-duplicate claims
2. Evidence quality prediction: Skip generation for low-quality evidence
3. Claim importance scoring: Prioritize high-value claims
"""

import logging
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cached result for a claim."""
    claim_text: str
    generated_text: str
    embedding: np.ndarray
    confidence: float
    evidence_fingerprint: str


class SemanticDeduplicationCache:
    """
    Cache LLM results using semantic similarity.
    
    If a new claim is semantically similar to a cached claim (>0.95 similarity),
    reuse the cached result instead of making a new LLM call.
    
    Expected speedup: 2-3 LLM calls saved per 10 claims (20-30% reduction).
    """
    
    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize semantic cache.
        
        Args:
            similarity_threshold: Cosine similarity threshold for cache hit
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.similarity_threshold = similarity_threshold
        self.hits = 0
        self.misses = 0
        
        logger.info(f"SemanticDeduplicationCache initialized (threshold={similarity_threshold})")
    
    def _compute_evidence_fingerprint(self, evidence_items: List[Any]) -> str:
        """Compute hash of evidence for cache validation."""
        evidence_texts = sorted([ev.snippet[:200] for ev in evidence_items])
        combined = "|".join(evidence_texts)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def lookup(
        self,
        claim_embedding: np.ndarray,
        evidence_items: List[Any]
    ) -> Optional[str]:
        """
        Look up cached result for semantically similar claim.
        
        Args:
            claim_embedding: Embedding of claim draft text
            evidence_items: Evidence items (for fingerprint validation)
        
        Returns:
            Cached generated text if hit, None if miss
        """
        if not self.cache:
            self.misses += 1
            return None
        
        evidence_fp = self._compute_evidence_fingerprint(evidence_items)
        
        # Find most similar cached claim
        max_similarity = 0.0
        best_entry = None
        
        for entry in self.cache.values():
            # Compare evidence fingerprints first (fast)
            if entry.evidence_fingerprint != evidence_fp:
                continue
            
            # Compute cosine similarity
            similarity = np.dot(claim_embedding, entry.embedding) / (
                np.linalg.norm(claim_embedding) * np.linalg.norm(entry.embedding)
            )
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_entry = entry
        
        # Cache hit if similarity above threshold
        if max_similarity >= self.similarity_threshold and best_entry:
            self.hits += 1
            logger.info(
                f"Cache HIT (similarity={max_similarity:.3f}): "
                f"Reusing result for '{best_entry.claim_text[:40]}...'"
            )
            return best_entry.generated_text
        
        self.misses += 1
        return None
    
    def store(
        self,
        claim_text: str,
        claim_embedding: np.ndarray,
        evidence_items: List[Any],
        generated_text: str,
        confidence: float
    ):
        """
        Store generated result in cache.
        
        Args:
            claim_text: Original claim draft text
            claim_embedding: Embedding of claim
            evidence_items: Evidence items used
            generated_text: LLM-generated definition
            confidence: Confidence score
        """
        evidence_fp = self._compute_evidence_fingerprint(evidence_items)
        cache_key = hashlib.sha256(claim_text.encode()).hexdigest()[:16]
        
        self.cache[cache_key] = CacheEntry(
            claim_text=claim_text,
            generated_text=generated_text,
            embedding=claim_embedding,
            confidence=confidence,
            evidence_fingerprint=evidence_fp
        )
        
        logger.debug(f"Cached result for '{claim_text[:40]}...' (key={cache_key})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "llm_calls_saved": self.hits
        }


class EvidenceQualityPredictor:
    """
    Predict if evidence is sufficient before expensive LLM call.
    
    Uses fast heuristics:
    - Max similarity score
    - Number of independent sources
    - Evidence length
    - Keyword overlap
    
    Expected speedup: Skip 30% of rejected claims early (saves ~70s).
    """
    
    def __init__(
        self,
        min_similarity: float = 0.25,  # Lowered from 0.4 to 0.25 for online Wikipedia evidence
        min_sources: int = 1,
        min_evidence_length: int = 30  # Lowered from 50 to 30 for shorter snippets
    ):
        """
        Initialize evidence quality predictor.
        
        Args:
            min_similarity: Minimum max similarity for quality (0.25 works well for Wikipedia)
            min_sources: Minimum independent sources
            min_evidence_length: Minimum total evidence length
        """
        self.min_similarity = min_similarity
        self.min_sources = min_sources
        self.min_evidence_length = min_evidence_length
        
        self.skipped_count = 0
        self.processed_count = 0
    
    def predict_quality(
        self,
        evidence_items: List[Any],
        claim_text: str
    ) -> Tuple[bool, str]:
        """
        Predict if evidence is sufficient for generation.
        
        Args:
            evidence_items: Retrieved evidence
            claim_text: Claim draft text
        
        Returns:
            (is_sufficient, reason) tuple
        """
        # Check 1: Any evidence retrieved?
        if not evidence_items:
            self.skipped_count += 1
            return False, "no_evidence"
        
        # Check 2: Similarity too low?
        similarities = [ev.similarity for ev in evidence_items if hasattr(ev, 'similarity')]
        max_sim = max(similarities) if similarities else 0.0
        
        if max_sim < self.min_similarity:
            self.skipped_count += 1
            return False, f"low_similarity_{max_sim:.2f}"
        
        # Check 3: Too few sources?
        unique_sources = len(set(ev.source_id for ev in evidence_items))
        if unique_sources < self.min_sources:
            self.skipped_count += 1
            return False, f"few_sources_{unique_sources}"
        
        # Check 4: Evidence too short?
        total_length = sum(len(ev.snippet) for ev in evidence_items)
        if total_length < self.min_evidence_length:
            self.skipped_count += 1
            return False, f"short_evidence_{total_length}"
        
        # Evidence looks good
        self.processed_count += 1
        return True, "sufficient"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        total = self.skipped_count + self.processed_count
        skip_rate = self.skipped_count / total if total > 0 else 0.0
        
        return {
            "skipped": self.skipped_count,
            "processed": self.processed_count,
            "skip_rate": skip_rate,
            "llm_calls_saved": self.skipped_count
        }


class ClaimPriorityScorer:
    """
    Score claim importance for adaptive processing.
    
    High-priority claims:
    - Core concepts (short, fundamental)
    - High-confidence evidence
    - Referenced by other claims
    
    Expected benefit: Process important claims first, abort low-priority if timeout.
    """
    
    def __init__(self):
        self.scored_count = 0
    
    def score_claim(
        self,
        claim_text: str,
        evidence_items: List[Any],
        is_referenced: bool = False
    ) -> float:
        """
        Compute priority score (0.0 to 1.0, higher = more important).
        
        Args:
            claim_text: Claim draft text
            evidence_items: Retrieved evidence
            is_referenced: True if other claims reference this one
        
        Returns:
            Priority score (0.0 - 1.0)
        """
        score = 0.0
        
        # Factor 1: Short claims are often core concepts (0.3 weight)
        length_score = max(0, 1.0 - len(claim_text) / 100.0)
        score += 0.3 * length_score
        
        # Factor 2: High evidence similarity (0.4 weight)
        if evidence_items:
            similarities = [ev.similarity for ev in evidence_items if hasattr(ev, 'similarity')]
            max_sim = max(similarities) if similarities else 0.0
            score += 0.4 * max_sim
        
        # Factor 3: Referenced by other claims (0.3 weight)
        if is_referenced:
            score += 0.3
        
        self.scored_count += 1
        return min(1.0, score)
    
    def rank_claims(
        self,
        claims: List[Any],
        evidence_map: Dict[int, List[Any]]
    ) -> List[Tuple[int, float]]:
        """
        Rank claims by priority.
        
        Args:
            claims: List of claims
            evidence_map: Mapping from claim index to evidence items
        
        Returns:
            List of (claim_index, priority_score) sorted by priority (descending)
        """
        scores = []
        for i, claim in enumerate(claims):
            evidence = evidence_map.get(i, [])
            score = self.score_claim(
                claim_text=claim.metadata.get("draft_text", claim.claim_text),
                evidence_items=evidence
            )
            scores.append((i, score))
        
        # Sort by score (high to low)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
