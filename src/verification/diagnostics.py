"""
Verification Pipeline Diagnostics

Provides structured instrumentation for debugging mass rejection issues:
- Claim-level debug logging
- Retrieval health checks
- NLI distribution analysis
- Chunking validation
- Session-level summaries
- JSON debug reports
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import Counter
from datetime import datetime
from pathlib import Path

import config
from src.claims.schema import LearningClaim, VerificationStatus, RejectionReason

logger = logging.getLogger(__name__)


@dataclass
class ClaimDebugInfo:
    """Debug information for a single claim."""
    claim_id: str
    claim_text: str
    retrieved_passage_count: int
    top_similarities: List[float]
    top_entailments: List[float]
    top_contradictions: List[float]
    independent_source_count: int
    final_status: str
    rejection_reason: Optional[str]


@dataclass
class RetrievalDiagnostics:
    """Retrieval health metrics."""
    max_similarity: float
    avg_similarity: float
    num_candidates: int
    is_empty: bool
    status: str  # "HEALTHY", "WEAK", "EMPTY"


@dataclass
class NLIDistribution:
    """NLI output probability distribution."""
    mean_entailment: float
    mean_neutral: float
    mean_contradiction: float
    max_entailment: float
    min_entailment: float


@dataclass
class SessionDiagnostics:
    """Session-level diagnostic summary."""
    session_id: str
    total_claims: int
    verified_count: int
    low_confidence_count: int
    rejected_count: int
    avg_max_similarity: float
    avg_max_entailment: float
    avg_max_contradiction: float
    avg_independent_sources: float
    rejection_reason_distribution: Dict[str, int]
    nli_distribution: Optional[NLIDistribution]
    mode: str  # "STANDARD" or "RELAXED"


class VerificationDiagnostics:
    """
    Central diagnostics coordinator for verification pipeline.
    
    Usage:
        diag = VerificationDiagnostics(session_id="session_123")
        
        # Log each claim
        diag.log_claim_verification(
            claim=claim_obj,
            retrieved_passages=5,
            similarities=[0.71, 0.65, 0.60],
            entailments=[0.72, 0.48, 0.31],
            contradictions=[0.05, 0.02],
            independent_sources=1,
            status=VerificationStatus.LOW_CONFIDENCE,
            reason=RejectionReason.INSUFFICIENT_SOURCES
        )
        
        # Log NLI distribution
        diag.log_nli_distribution(nli_results)
        
        # Get session summary
        summary = diag.get_session_summary()
    """
    
    def __init__(self, session_id: str):
        """
        Initialize diagnostics tracker.
        
        Args:
            session_id: Session identifier for grouping diagnostics
        """
        self.session_id = session_id
        self.claims: List[ClaimDebugInfo] = []
        self.retrieval_diagnostics: List[RetrievalDiagnostics] = []
        self.nli_distributions: List[NLIDistribution] = []
        self.rejection_reason_counts: Counter = Counter()
        self.status_counts: Dict[str, int] = {
            VerificationStatus.VERIFIED: 0,
            VerificationStatus.LOW_CONFIDENCE: 0,
            VerificationStatus.REJECTED: 0
        }
        self.similarities: List[float] = []
        self.entailments: List[float] = []
        self.contradictions: List[float] = []
        self.source_counts: List[int] = []
        
    def log_claim_verification(
        self,
        claim: LearningClaim,
        retrieved_passages: int,
        similarities: List[float],
        entailments: List[float],
        contradictions: List[float],
        independent_sources: int,
        status: VerificationStatus,
        reason: Optional[RejectionReason] = None
    ) -> None:
        """
        Log verification data for a single claim.
        
        Args:
            claim: Learning claim object
            retrieved_passages: Number of retrieved evidence passages
            similarities: List of similarity scores (top-k)
            entailments: List of entailment probabilities (top-k)
            contradictions: List of contradiction probabilities (top-k)
            independent_sources: Count of independent sources
            status: Final verification status
            reason: Rejection reason if rejected/low_confidence
        """
        debug_info = ClaimDebugInfo(
            claim_id=claim.claim_id[:16] if hasattr(claim, 'claim_id') else "unknown",
            claim_text=claim.claim_text[:100] if hasattr(claim, 'claim_text') else "unknown",
            retrieved_passage_count=retrieved_passages,
            top_similarities=similarities[:5],
            top_entailments=entailments[:5],
            top_contradictions=contradictions[:5],
            independent_source_count=independent_sources,
            final_status=status,
            rejection_reason=reason.value if reason else None
        )
        
        self.claims.append(debug_info)
        self.status_counts[status] += 1
        if reason:
            self.rejection_reason_counts[reason.value] += 1
        
        # Track aggregate metrics
        if similarities:
            self.similarities.append(max(similarities))
        if entailments:
            self.entailments.append(max(entailments))
        if contradictions:
            self.contradictions.append(max(contradictions))
        self.source_counts.append(independent_sources)
        
        # Print debug info if enabled
        if config.DEBUG_VERIFICATION:
            self._print_claim_debug(debug_info)
    
    def log_retrieval_diagnostics(
        self,
        max_similarity: float,
        avg_similarity: float,
        num_candidates: int,
        warning_threshold: float = 0.45
    ) -> None:
        """
        Log retrieval health check results.
        
        Args:
            max_similarity: Maximum similarity score
            avg_similarity: Average similarity across candidates
            num_candidates: Number of retrieved candidates
            warning_threshold: Threshold below which to warn
        """
        is_empty = num_candidates == 0
        is_weak = max_similarity < warning_threshold and num_candidates > 0
        
        status = "EMPTY" if is_empty else ("WEAK" if is_weak else "HEALTHY")
        
        diag = RetrievalDiagnostics(
            max_similarity=max_similarity,
            avg_similarity=avg_similarity,
            num_candidates=num_candidates,
            is_empty=is_empty,
            status=status
        )
        
        self.retrieval_diagnostics.append(diag)
        
        if config.DEBUG_RETRIEVAL_HEALTH and (is_weak or is_empty):
            logger.warning(
                f"Retrieval health: {status} - "
                f"max_sim={max_similarity:.3f}, avg_sim={avg_similarity:.3f}, "
                f"candidates={num_candidates}"
            )
    
    def log_nli_distribution(self, nli_results: List[Dict[str, Any]]) -> None:
        """
        Log and analyze NLI output probability distribution.
        
        Args:
            nli_results: List of NLI classification results
                        Format: [{"label": "entailment|neutral|contradiction", "score": float}, ...]
        """
        if not nli_results:
            return
        
        entailment_scores = [r.get("score", 0.0) for r in nli_results if r.get("label") == "entailment"]
        neutral_scores = [r.get("score", 0.0) for r in nli_results if r.get("label") == "neutral"]
        contradiction_scores = [r.get("score", 0.0) for r in nli_results if r.get("label") == "contradiction"]
        
        dist = NLIDistribution(
            mean_entailment=sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0.0,
            mean_neutral=sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0.0,
            mean_contradiction=sum(contradiction_scores) / len(contradiction_scores) if contradiction_scores else 0.0,
            max_entailment=max(entailment_scores) if entailment_scores else 0.0,
            min_entailment=min(entailment_scores) if entailment_scores else 0.0
        )
        
        self.nli_distributions.append(dist)
        
        if config.DEBUG_NLI_DISTRIBUTION:
            if dist.mean_entailment < 0.40:
                logger.warning(
                    f"NLI distribution skewed: mean_entailment={dist.mean_entailment:.3f} < 0.40 - "
                    f"possible chunking/alignment mismatch. "
                    f"(neutral={dist.mean_neutral:.3f}, contradiction={dist.mean_contradiction:.3f})"
                )
            else:
                logger.info(
                    f"NLI distribution: entailment={dist.mean_entailment:.3f}, "
                    f"neutral={dist.mean_neutral:.3f}, contradiction={dist.mean_contradiction:.3f}"
                )
    
    def log_chunking_validation(
        self,
        total_source_length: int,
        num_chunks: int,
        avg_chunk_size: float,
        warning_thresholds: tuple = (800, 50)
    ) -> None:
        """
        Validate source chunking and warn if problematic.
        
        Args:
            total_source_length: Total characters in source material
            num_chunks: Number of chunks created
            avg_chunk_size: Average chunk size in tokens/characters
            warning_thresholds: (too_large, too_small) thresholds
        """
        too_large, too_small = warning_thresholds
        
        if config.DEBUG_CHUNKING:
            logger.info(
                f"Chunking: {num_chunks} chunks, "
                f"avg_size={avg_chunk_size:.0f}, "
                f"total_length={total_source_length}"
            )
            
            if avg_chunk_size > too_large:
                logger.warning(
                    f"Chunk size too large ({avg_chunk_size:.0f} > {too_large}): "
                    f"May harm NLI alignment and evidence matching."
                )
            elif avg_chunk_size < too_small:
                logger.warning(
                    f"Chunk size too small ({avg_chunk_size:.0f} < {too_small}): "
                    f"May lose context for evidence matching."
                )
    
    def get_session_summary(self, mode: str = "STANDARD") -> SessionDiagnostics:
        """
        Compute session-level diagnostic summary.
        
        Args:
            mode: "STANDARD" or "RELAXED" verification mode
        
        Returns:
            SessionDiagnostics with aggregated metrics
        """
        summary = SessionDiagnostics(
            session_id=self.session_id,
            total_claims=len(self.claims),
            verified_count=self.status_counts.get(VerificationStatus.VERIFIED, 0),
            low_confidence_count=self.status_counts.get(VerificationStatus.LOW_CONFIDENCE, 0),
            rejected_count=self.status_counts.get(VerificationStatus.REJECTED, 0),
            avg_max_similarity=sum(self.similarities) / len(self.similarities) if self.similarities else 0.0,
            avg_max_entailment=sum(self.entailments) / len(self.entailments) if self.entailments else 0.0,
            avg_max_contradiction=sum(self.contradictions) / len(self.contradictions) if self.contradictions else 0.0,
            avg_independent_sources=sum(self.source_counts) / len(self.source_counts) if self.source_counts else 0.0,
            rejection_reason_distribution=dict(self.rejection_reason_counts),
            nli_distribution=self._aggregate_nli_distribution(),
            mode=mode
        )
        
        return summary
    
    def print_session_summary(self, summary: SessionDiagnostics) -> None:
        """Pretty-print session diagnostics summary."""
        print("\n" + "="*70)
        print("SESSION DIAGNOSTICS")
        print("="*70)
        print(f"Session ID: {summary.session_id}")
        print(f"Mode: {summary.mode}")
        print(f"\nClaim Status Breakdown:")
        print(f"  Total claims: {summary.total_claims}")
        print(f"  Verified: {summary.verified_count} ({100*summary.verified_count/max(summary.total_claims, 1):.1f}%)")
        print(f"  Low Confidence: {summary.low_confidence_count} ({100*summary.low_confidence_count/max(summary.total_claims, 1):.1f}%)")
        print(f"  Rejected: {summary.rejected_count} ({100*summary.rejected_count/max(summary.total_claims, 1):.1f}%)")
        
        print(f"\nAggregated Metrics:")
        print(f"  Avg max similarity: {summary.avg_max_similarity:.3f}")
        print(f"  Avg max entailment: {summary.avg_max_entailment:.3f}")
        print(f"  Avg max contradiction: {summary.avg_max_contradiction:.3f}")
        print(f"  Avg independent sources: {summary.avg_independent_sources:.2f}")
        
        if summary.rejection_reason_distribution:
            print(f"\nTop Rejection Reasons:")
            for reason, count in sorted(
                summary.rejection_reason_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                print(f"  {reason}: {count}")
        
        if summary.nli_distribution:
            print(f"\nNLI Distribution:")
            print(f"  Mean entailment: {summary.nli_distribution.mean_entailment:.3f}")
            print(f"  Mean neutral: {summary.nli_distribution.mean_neutral:.3f}")
            print(f"  Mean contradiction: {summary.nli_distribution.mean_contradiction:.3f}")
        
        print("="*70 + "\n")
    
    def save_debug_report(self, output_dir: Path = None) -> Path:
        """
        Save detailed debug report as JSON.
        
        Args:
            output_dir: Output directory (defaults to outputs/)
        
        Returns:
            Path to saved JSON file
        """
        if output_dir is None:
            output_dir = config.OUTPUT_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare report
        summary = self.get_session_summary()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "mode": "RELAXED" if config.RELAXED_VERIFICATION_MODE else "STANDARD",
            "summary": {
                "total_claims": summary.total_claims,
                "verified": summary.verified_count,
                "low_confidence": summary.low_confidence_count,
                "rejected": summary.rejected_count,
                "avg_similarity": summary.avg_max_similarity,
                "avg_entailment": summary.avg_max_entailment,
                "avg_contradiction": summary.avg_max_contradiction,
                "avg_sources": summary.avg_independent_sources
            },
            "rejection_reasons": summary.rejection_reason_distribution,
            "nli_distribution": asdict(summary.nli_distribution) if summary.nli_distribution else None,
            "config_thresholds": {
                "min_entailment_prob": config.MIN_ENTAILMENT_PROB,
                "min_supporting_sources": config.MIN_SUPPORTING_SOURCES,
                "max_contradiction_prob": config.MAX_CONTRADICTION_PROB
            },
            "claims": [asdict(c) for c in self.claims[:20]]  # First 20 claims for inspection
        }
        
        report_path = output_dir / f"debug_session_report_{self.session_id[:12]}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Debug report saved to {report_path}")
        return report_path
    
    # ======================== PRIVATE HELPERS ========================
    
    def _print_claim_debug(self, debug: ClaimDebugInfo) -> None:
        """Pretty-print claim debug information."""
        print("\n" + "-"*70)
        print("CLAIM DEBUG")
        print(f"ID: {debug.claim_id}")
        print(f"TEXT: {debug.claim_text[:120]}...")
        print(f"Retrieved passages: {debug.retrieved_passage_count}")
        print(f"Top similarities: {[f'{s:.2f}' for s in debug.top_similarities]}")
        print(f"Top entailments: {[f'{e:.2f}' for e in debug.top_entailments]}")
        print(f"Top contradictions: {[f'{c:.2f}' for c in debug.top_contradictions]}")
        print(f"Independent sources: {debug.independent_source_count}")
        print(f"Decision: {debug.final_status}")
        if debug.rejection_reason:
            print(f"Reason: {debug.rejection_reason}")
        print("-"*70)
    
    def _aggregate_nli_distribution(self) -> Optional[NLIDistribution]:
        """Average all logged NLI distributions."""
        if not self.nli_distributions:
            return None
        
        n = len(self.nli_distributions)
        return NLIDistribution(
            mean_entailment=sum(d.mean_entailment for d in self.nli_distributions) / n,
            mean_neutral=sum(d.mean_neutral for d in self.nli_distributions) / n,
            mean_contradiction=sum(d.mean_contradiction for d in self.nli_distributions) / n,
            max_entailment=max(d.max_entailment for d in self.nli_distributions),
            min_entailment=min(d.min_entailment for d in self.nli_distributions)
        )
