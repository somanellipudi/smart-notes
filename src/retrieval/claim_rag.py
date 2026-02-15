"""
Evidence-First Retrieval for Claim Validation

This module implements the evidence retrieval and validation pipeline:
1. Retrieve candidate evidence for each claim
2. Compute similarity scores
3. Enforce independence (multiple sources)
4. Compute consistency (semantic agreement)
5. Detect conflicts (contradictions)
6. Apply hybrid decision policy

See docs/VERIFIABILITY_CONTRACT.md for formal specification.
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from src.claims.schema import LearningClaim, EvidenceItem, RejectionReason, VerificationStatus

logger = logging.getLogger(__name__)


class ClaimRAG:
    """Evidence-first retrieval and validation for learning claims."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RAG with configuration thresholds."""
        self.config = config or {}
        self.tau = self.config.get("tau", 0.2)
        self.k = self.config.get("k", 1)
        self.sigma_min = self.config.get("sigma_min", 0.7)
        self.t_verify = self.config.get("t_verify", 0.5)
        self.t_reject = self.config.get("t_reject", 0.2)
        self.min_evidence_length = self.config.get("min_evidence_length", 15)
    
    def _extract_key_terms(self, text: str, max_terms: int = 5) -> List[str]:
        """Extract key terms from text for matching."""
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = {'the', 'is', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'not', 'that', 'this'}
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        return keywords[:max_terms]
    
    def _find_matching_spans(
        self,
        source: str,
        key_terms: List[str],
        search_text: str,
        span_length: int = 150
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Find text spans in source that match key terms."""
        matches = []
        
        for term in key_terms:
            for match in re.finditer(re.escape(term), source, re.IGNORECASE):
                start_pos = max(0, match.start() - span_length // 2)
                end_pos = min(len(source), match.end() + span_length // 2)
                
                span = source[start_pos:end_pos].strip()
                
                if len(span) < self.min_evidence_length:
                    continue
                
                metadata = {"start_char": start_pos, "end_char": end_pos, "matched_term": term}
                matches.append((span, metadata))
        
        unique_matches = []
        for span, meta in matches:
            is_duplicate = any(
                self._compute_similarity(span, existing_span) > 0.9
                for existing_span, _ in unique_matches
            )
            if not is_duplicate:
                unique_matches.append((span, meta))
        
        return unique_matches
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts (0-1)."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0.0
        length_bonus = min(len(text2) / (len(text1) + 1) * 0.2, 0.1)
        
        return min(jaccard + length_bonus, 1.0)
    
    def retrieve_evidence_for_claim(
        self,
        claim: LearningClaim,
        sources: List[str]
    ) -> List[EvidenceItem]:
        """Retrieve evidence for a claim from source materials."""
        evidence_list = []
        search_text = claim.metadata.get("draft_text", "") or claim.claim_text
        
        if not search_text or len(search_text) < 3:
            logger.warning(f"Claim {claim.claim_id} has no search text")
            return []
        
        key_terms = self._extract_key_terms(search_text)
        
        for source_idx, source in enumerate(sources):
            if not source or len(source) < 10:
                continue
            
            matches = self._find_matching_spans(source, key_terms, search_text)
            
            for span_text, span_metadata in matches:
                similarity = self._compute_similarity(search_text, span_text)
                
                if similarity >= self.tau:
                    evidence = EvidenceItem(
                        source_id=f"source_{source_idx}",
                        source_type="notes",
                        snippet=span_text,
                        span_metadata=span_metadata,
                        similarity=similarity,
                        reliability_prior=0.8
                    )
                    evidence_list.append(evidence)
        
        evidence_list.sort(key=lambda e: e.similarity, reverse=True)
        logger.debug(f"Claim {claim.claim_id}: retrieved {len(evidence_list)} evidence items")
        return evidence_list
    
    def enforce_independence(
        self,
        evidence: List[EvidenceItem],
        k: Optional[int] = None
    ) -> List[EvidenceItem]:
        """Filter evidence to enforce independence (k independent sources)."""
        if k is None:
            k = self.k
        
        if len(evidence) < k:
            return evidence
        
        independent = []
        used_sources = set()
        
        for item in evidence:
            source_key = (item.source_id, item.span_metadata.get("start_char", 0))
            if source_key not in used_sources:
                independent.append(item)
                used_sources.add(source_key)
                if len(independent) >= k:
                    break
        
        return independent
    
    def compute_consistency(self, evidence: List[EvidenceItem]) -> float:
        """Compute semantic consistency score across evidence (0-1)."""
        if len(evidence) < 2:
            return 1.0
        
        consistency_scores = []
        
        for i in range(len(evidence)):
            for j in range(i + 1, len(evidence)):
                snippet1 = evidence[i].snippet.lower()
                snippet2 = evidence[j].snippet.lower()
                
                contradictions = [('not', ''), ('never', 'always'), ('false', 'true'), ('wrong', 'right')]
                
                has_contradiction = False
                for neg, pos in contradictions:
                    if neg in snippet1 and pos in snippet2:
                        has_contradiction = True
                    if neg in snippet2 and pos in snippet1:
                        has_contradiction = True
                
                sim = self._compute_similarity(snippet1, snippet2)
                
                if has_contradiction and sim > 0.5:
                    consistency_scores.append(0.0)
                elif has_contradiction:
                    consistency_scores.append(0.3)
                elif sim > 0.7:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.7)
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0
        return min(avg_consistency, 1.0)
    
    def detect_conflicts(self, evidence: List[EvidenceItem]) -> bool:
        """Detect explicit contradictions in evidence."""
        if len(evidence) < 2:
            return False
        
        contradiction_pairs = [('always', 'never'), ('true', 'false'), ('yes', 'no'), ('increase', 'decrease')]
        
        for i in range(len(evidence)):
            for j in range(i + 1, len(evidence)):
                text1 = evidence[i].snippet.lower()
                text2 = evidence[j].snippet.lower()
                
                for pos, neg in contradiction_pairs:
                    if pos in text1 and neg in text2:
                        if self._compute_similarity(text1, text2) > 0.6:
                            return True
        
        return False
    
    def apply_decision_policy(
        self,
        claim: LearningClaim,
        evidence: List[EvidenceItem],
        custom_config: Optional[Dict[str, float]] = None
    ) -> Tuple[VerificationStatus, Optional[RejectionReason], float]:
        """Apply hybrid decision policy to determine claim status."""
        config = {**self.config, **(custom_config or {})}
        tau = config.get("tau", self.tau)
        k = config.get("k", self.k)
        sigma_min = config.get("sigma_min", self.sigma_min)
        t_verify = config.get("t_verify", self.t_verify)
        t_reject = config.get("t_reject", self.t_reject)
        
        if not evidence:
            return VerificationStatus.REJECTED, RejectionReason.NO_EVIDENCE, 0.0
        
        best_similarity = max(e.similarity for e in evidence)
        if best_similarity < tau:
            return VerificationStatus.REJECTED, RejectionReason.LOW_SIMILARITY, best_similarity
        
        independent_evidence = self.enforce_independence(evidence, k)
        if len(independent_evidence) < k:
            return VerificationStatus.LOW_CONFIDENCE, RejectionReason.INSUFFICIENT_SOURCES, best_similarity
        
        consistency = self.compute_consistency(evidence)
        if consistency < sigma_min:
            return VerificationStatus.LOW_CONFIDENCE, RejectionReason.LOW_CONSISTENCY, consistency
        
        if self.detect_conflicts(evidence):
            return VerificationStatus.LOW_CONFIDENCE, RejectionReason.CONFLICT, best_similarity
        
        avg_similarity = sum(e.similarity * e.reliability_prior for e in independent_evidence) / len(independent_evidence)
        redundancy_bonus = min(0.1 * (len(independent_evidence) - 1), 0.15)
        confidence = min(avg_similarity * (1 + redundancy_bonus), 1.0)
        
        if confidence >= t_verify:
            return VerificationStatus.VERIFIED, None, confidence
        elif confidence >= t_reject:
            return VerificationStatus.LOW_CONFIDENCE, None, confidence
        else:
            return VerificationStatus.REJECTED, RejectionReason.INSUFFICIENT_CONFIDENCE, confidence
    
    def process_claims(
        self,
        claims: List[LearningClaim],
        sources: List[str]
    ) -> List[LearningClaim]:
        """Process multiple claims with evidence retrieval and validation."""
        processed = []
        
        for claim in claims:
            evidence = self.retrieve_evidence_for_claim(claim, sources)
            claim.evidence_objects = evidence
            
            status, rejection_reason, confidence = self.apply_decision_policy(claim, evidence)
            
            claim.status = status
            claim.rejection_reason = rejection_reason
            claim.confidence = confidence
            
            processed.append(claim)
        
        return processed
