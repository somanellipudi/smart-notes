"""
Verification diagnostics and rejection analysis module.

Provides detailed logging of claim verification results, rejection reasons,
and evidence retrieval metrics for debugging and analysis.
"""

import logging
from typing import Dict, List, Any
from collections import defaultdict
from src.claims.schema import VerificationStatus

logger = logging.getLogger(__name__)


class RejectionHistogram:
    """
    Tracks claim rejection reasons and evidence retrieval metrics.
    
    Provides histograms of rejection counts by reason and breakdown
    of retrieval hit rates for analysis.
    """
    
    def __init__(self):
        """Initialize rejection tracking."""
        self.rejection_counts: Dict[str, int] = defaultdict(int)
        self.retrieval_hits: List[int] = []  # Number of hits per claim
        self.total_claims = 0
        self.verified_claims = 0
        self.rejected_claims = 0
        self.uncertain_claims = 0
    
    def add_claim_result(
        self,
        claim_text: str,
        status: VerificationStatus,
        rejection_reason: str = None,
        retrieval_hit_count: int = 0
    ) -> None:
        """
        Record verification result for a claim.
        
        Args:
            claim_text: The claim text (for logging)
            status: VerificationStatus enum
            rejection_reason: Specific reason for rejection (if applicable)
            retrieval_hit_count: Number of evidence hits retrieved for this claim
        """
        self.total_claims += 1
        self.retrieval_hits.append(retrieval_hit_count)
        
        if status == VerificationStatus.VERIFIED:
            self.verified_claims += 1
        elif status == VerificationStatus.REJECTED:
            self.rejected_claims += 1
            if rejection_reason:
                self.rejection_counts[rejection_reason] += 1
        elif status == VerificationStatus.UNCERTAIN:
            self.uncertain_claims += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dict with verification counts and metrics
        """
        return {
            'total_claims': self.total_claims,
            'verified': self.verified_claims,
            'rejected': self.rejected_claims,
            'uncertain': self.uncertain_claims,
            'verified_rate': self.verified_claims / max(1, self.total_claims),
            'rejected_rate': self.rejected_claims / max(1, self.total_claims),
            'rejection_reasons': dict(sorted(
                self.rejection_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'avg_retrieval_hits': sum(self.retrieval_hits) / max(1, len(self.retrieval_hits)),
            'max_retrieval_hits': max(self.retrieval_hits) if self.retrieval_hits else 0,
            'min_retrieval_hits': min(self.retrieval_hits) if self.retrieval_hits else 0,
            'zero_hit_claims': sum(1 for h in self.retrieval_hits if h == 0)
        }
    
    def log_summary(self) -> None:
        """Log rejection histogram summary."""
        summary = self.get_summary()
        
        logger.info("=" * 60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total claims: {summary['total_claims']}")
        logger.info(f"  ✓ Verified: {summary['verified']} ({summary['verified_rate']:.1%})")
        logger.info(f"  ✗ Rejected: {summary['rejected']} ({summary['rejected_rate']:.1%})")
        logger.info(f"  ? Uncertain: {summary['uncertain']}")
        
        if summary['rejection_reasons']:
            logger.info("\nRejection Reasons (Top 5):")
            for reason, count in list(summary['rejection_reasons'].items())[:5]:
                logger.info(f"  - {reason}: {count}")
        
        logger.info(f"\nEvidence Retrieval Metrics:")
        logger.info(f"  Avg hits per claim: {summary['avg_retrieval_hits']:.2f}")
        logger.info(f"  Max hits: {summary['max_retrieval_hits']}")
        logger.info(f"  Claims with zero hits: {summary['zero_hit_claims']}")
        logger.info("=" * 60)


class VerificationDebugMetadata:
    """
    Captures debug metadata for verification response.
    
    Tracks evidence store status, retrieval performance, and claim statistics
    for inclusion in response JSON.
    """
    
    def __init__(self):
        """Initialize debug metadata."""
        self.evidence_docs_count = 0
        self.evidence_chunks_count = 0
        self.avg_chunk_length = 0.0
        self.claims_count = 0
        self.retrieval_hit_rate = 0.0
        self.embedding_method = "unknown"
        self.quality_issues = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'evidence_docs_count': self.evidence_docs_count,
            'evidence_chunks_count': self.evidence_chunks_count,
            'avg_chunk_length': round(self.avg_chunk_length, 2),
            'claims_count': self.claims_count,
            'retrieval_hit_rate': round(self.retrieval_hit_rate, 3),
            'embedding_method': self.embedding_method,
            'quality_issues': self.quality_issues
        }


def log_rejection_histogram(histogram: RejectionHistogram) -> Dict[str, Any]:
    """
    Log and return rejection histogram summary.
    
    Args:
        histogram: RejectionHistogram instance
        
    Returns:
        Summary dict for logging/response
    """
    summary = histogram.get_summary()
    histogram.log_summary()
    return summary


def create_verification_response_metadata(
    histogram: RejectionHistogram,
    evidence_store_stats: Dict[str, Any],
    embedding_method: str = "unknown"
) -> VerificationDebugMetadata:
    """
    Create debug metadata for verification response.
    
    Args:
        histogram: RejectionHistogram with verification results
        evidence_store_stats: Evidence store statistics dict
        embedding_method: Name of embedding method used
        
    Returns:
        VerificationDebugMetadata instance
    """
    metadata = VerificationDebugMetadata()
    
    # Set from evidence store
    metadata.evidence_docs_count = evidence_store_stats.get('num_sources', 0)
    metadata.evidence_chunks_count = evidence_store_stats.get('num_chunks', 0)
    
    if metadata.evidence_chunks_count > 0:
        total_chars = evidence_store_stats.get('total_chars', 0)
        metadata.avg_chunk_length = total_chars / metadata.evidence_chunks_count
    
    # Set from verification histogram
    metadata.claims_count = histogram.total_claims
    summary = histogram.get_summary()
    
    if histogram.total_claims > 0:
        zero_hits = summary.get('zero_hit_claims', 0)
        metadata.retrieval_hit_rate = 1.0 - (zero_hits / histogram.total_claims)
    
    metadata.embedding_method = embedding_method
    
    return metadata
