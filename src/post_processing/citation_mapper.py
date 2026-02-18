"""
Citation Mapping

Maps verification results (verified claims with supporting spans) to citations
in the output schema. Handles config-based filtering and claim-to-citation mapping.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from src.schema.output_schema import Citation

logger = logging.getLogger(__name__)


@dataclass
class VerifiedSpan:
    """Represents a verified evidence span."""
    span_id: str
    source_id: str
    source_type: str  # "local" or "online"
    snippet: str
    page_num: Optional[int] = None
    authority_tier: Optional[str] = None
    confidence: float = 1.0  # Verification confidence


class CitationMapper:
    """Map verification results to output schema citations."""
    
    def __init__(
        self,
        enable_citations: bool = True,
        show_unverified_with_label: bool = True,
        show_unverified_omit: bool = False,
        citation_max_per_claim: int = 3,
        require_citations_for_cs_claims: bool = True
    ):
        """
        Initialize citation mapper with configuration.
        
        Args:
            enable_citations: Enable citation processing
            show_unverified_with_label: Add "(needs evidence)" to unsupported claims
            show_unverified_omit: Omit unsupported claims entirely
            citation_max_per_claim: Max citations per claim
            require_citations_for_cs_claims: Require citations for CS claim types
        """
        self.enable_citations = enable_citations
        self.show_unverified_with_label = show_unverified_with_label
        self.show_unverified_omit = show_unverified_omit
        self.citation_max_per_claim = citation_max_per_claim
        self.require_citations_for_cs_claims = require_citations_for_cs_claims
        
        # Validate config
        if show_unverified_with_label and show_unverified_omit:
            raise ValueError(
                "Cannot set both show_unverified_with_label and show_unverified_omit"
            )
    
    def map_claim_to_citations(
        self,
        claim: str,
        verified_spans: List[VerifiedSpan],
        claim_type: Optional[str] = None
    ) -> Tuple[Optional[str], List[Citation]]:
        """
        Map a single claim to citations.
        
        Args:
            claim: The claim text
            verified_spans: List of verified evidence spans supporting claim
            claim_type: Optional claim type (e.g., COMPLEXITY_CLAIM, CODE_BEHAVIOR_CLAIM)
        
        Returns:
            Tuple of (modified_claim_text, citations_list)
            - modified_claim_text: May include "(needs evidence)" if configured
            - citations_list: List of Citation objects
        """
        if not self.enable_citations:
            return claim, []
        
        # Convert spans to citations
        citations = self._spans_to_citations(verified_spans)
        
        # Apply limit
        if len(citations) > self.citation_max_per_claim:
            citations = citations[:self.citation_max_per_claim]
        
        # Check if citation required for claim type
        if claim_type and self.require_citations_for_cs_claims:
            cs_types = {"COMPLEXITY_CLAIM", "CODE_BEHAVIOR_CLAIM", "DEFINITION_CLAIM", "NUMERIC_CLAIM"}
            if claim_type in cs_types and not citations:
                if self.show_unverified_omit:
                    return None, []  # Omit claim
                elif self.show_unverified_with_label:
                    claim = f"{claim} (needs evidence)"
        elif not citations:
            # No citations found
            if self.show_unverified_omit:
                return None, []  # Omit claim
            elif self.show_unverified_with_label:
                claim = f"{claim} (needs evidence)"
        
        return claim, citations
    
    def map_claims_to_citations(
        self,
        claims: List[str],
        claim_verification_map: Dict[str, List[VerifiedSpan]],
        claim_type_map: Optional[Dict[str, str]] = None
    ) -> Tuple[List[str], List[List[Citation]]]:
        """
        Map multiple claims to citations.
        
        Args:
            claims: List of claims
            claim_verification_map: Dict mapping claim -> List[VerifiedSpan]
            claim_type_map: Optional dict mapping claim -> claim_type
        
        Returns:
            Tuple of (filtered_claims, citations_per_claim)
            - filtered_claims: Claims after filtering (may be shorter if omitting unverified)
            - citations_per_claim: Corresponding citations for each claim
        """
        filtered_claims = []
        citations_per_claim = []
        
        for claim in claims:
            verified_spans = claim_verification_map.get(claim, [])
            claim_type = claim_type_map.get(claim) if claim_type_map else None
            
            modified_claim, citations = self.map_claim_to_citations(
                claim,
                verified_spans,
                claim_type
            )
            
            if modified_claim is not None:  # None means omit
                filtered_claims.append(modified_claim)
                citations_per_claim.append(citations)
        
        return filtered_claims, citations_per_claim
    
    @staticmethod
    def _spans_to_citations(spans: List[VerifiedSpan]) -> List[Citation]:
        """Convert verified spans to citations."""
        citations = []
        
        for span in spans:
            citation = Citation(
                span_id=span.span_id,
                source_id=span.source_id,
                source_type=span.source_type,
                snippet=span.snippet,
                page_num=span.page_num,
                authority_tier=span.authority_tier
            )
            citations.append(citation)
        
        return citations
    
    @staticmethod
    def batch_map_claims_to_citations(
        claim_batches: List[List[str]],
        verification_results: Dict,
        config: Optional[Dict] = None
    ) -> List[Tuple[List[List[Citation]]]]:
        """
        Map claims across multiple batches.
        
        Args:
            claim_batches: List of claim lists to process
            verification_results: Global verification results
            config: Optional configuration overrides
        
        Returns:
            List of citation lists per batch
        """
        mapper = CitationMapper(**(config or {}))
        results = []
        
        for claims in claim_batches:
            # Extract verification results for this batch
            claim_verification_map = {}
            for claim in claims:
                if claim in verification_results:
                    claim_verification_map[claim] = verification_results[claim]
            
            _, citations_per_claim = mapper.map_claims_to_citations(
                claims,
                claim_verification_map
            )
            results.append(citations_per_claim)
        
        return results
    
    @staticmethod
    def filter_citations_by_authority(
        citations: List[Citation],
        min_tier: Optional[str] = None,
        source_types: Optional[Set[str]] = None
    ) -> List[Citation]:
        """
        Filter citations by authority tier and source type.
        
        Args:
            citations: List of citations
            min_tier: Minimum authority tier (TIER_1, TIER_2, TIER_3)
            source_types: Set of allowed source types (local, online)
        
        Returns:
            Filtered citations
        """
        filtered = citations
        
        # Filter by tier
        if min_tier:
            tier_order = {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3}
            min_tier_val = tier_order.get(min_tier, 999)
            filtered = [
                c for c in filtered
                if not c.authority_tier or tier_order.get(c.authority_tier, 999) <= min_tier_val
            ]
        
        # Filter by source type
        if source_types:
            filtered = [c for c in filtered if c.source_type in source_types]
        
        return filtered
    
    @staticmethod
    def dedup_citations(citations: List[Citation]) -> List[Citation]:
        """
        Remove duplicate citations (same source_id and snippet).
        
        Args:
            citations: List of citations
        
        Returns:
            Deduplicated citations
        """
        seen: Set[Tuple] = set()
        deduped = []
        
        for citation in citations:
            key = (citation.span_id, citation.source_id, citation.snippet[:50] if citation.snippet else "")
            if key not in seen:
                seen.add(key)
                deduped.append(citation)
        
        return deduped
    
    @staticmethod
    def rank_citations_by_tier(citations: List[Citation]) -> List[Citation]:
        """
        Sort citations by authority tier (TIER_1 > TIER_2 > TIER_3 > None).
        
        Args:
            citations: List of citations
        
        Returns:
            Sorted citations
        """
        tier_order = {"TIER_1": 0, "TIER_2": 1, "TIER_3": 2, None: 3}
        return sorted(
            citations,
            key=lambda c: (tier_order.get(c.authority_tier, 999), c.source_id)
        )
    
    @staticmethod
    def get_citation_summary(citations: List[Citation]) -> Dict:
        """
        Generate summary statistics for citations.
        
        Args:
            citations: List of citations
        
        Returns:
            Summary dict
        """
        if not citations:
            return {
                "total": 0,
                "by_source": {},
                "by_tier": {},
                "by_type": {},
            }
        
        summary = {
            "total": len(citations),
            "by_source": {},
            "by_tier": {},
            "by_type": {},
        }
        
        for citation in citations:
            # By source
            summary["by_source"][citation.source_id] = summary["by_source"].get(citation.source_id, 0) + 1
            
            # By tier
            tier = citation.authority_tier or "unknown"
            summary["by_tier"][tier] = summary["by_tier"].get(tier, 0) + 1
            
            # By type
            src_type = citation.source_type
            summary["by_type"][src_type] = summary["by_type"].get(src_type, 0) + 1
        
        return summary


# Convenience function for quick mapping
def map_claims_to_citations(
    claims: List[str],
    verification_map: Dict[str, List[VerifiedSpan]],
    config: Optional[Dict] = None
) -> Tuple[List[str], List[List[Citation]]]:
    """
    Convenience function to map claims to citations.
    
    Args:
        claims: List of claims to map
        verification_map: Verification results dict
        config: Optional configuration dict
    
    Returns:
        Tuple of (filtered_claims, citations_per_claim)
    """
    mapper = CitationMapper(**(config or {}))
    return mapper.map_claims_to_citations(claims, verification_map)
