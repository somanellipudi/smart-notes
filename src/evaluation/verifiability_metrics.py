"""
Verifiability-specific evaluation metrics.

Measures rejection rates, unsupported claims, hallucination reduction,
and refusal rates for Verifiable Mode.

Research-grade metrics:
- Rejection reasons breakdown
- Traceability rate (% claims with evidence)
- Conflict rate
- Negative control detection (no evidence scenarios)
- Input sufficiency warnings
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.claims.schema import ClaimCollection, VerificationStatus
from src.graph.claim_graph import ClaimGraph

logger = logging.getLogger(__name__)


class VerifiabilityMetrics:
    """
    Metrics for evaluating verifiable mode performance.
    
    Tracks:
    - Rejection rates
    - Unsupported claim rates
    - Evidence quality
    - Refusal behavior
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics_history = []
        logger.info("VerifiabilityMetrics initialized")
    
    def calculate_metrics(
        self,
        claim_collection: ClaimCollection,
        graph_metrics: Optional[Dict[str, Any]] = None,
        baseline_output: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive verifiability metrics.
        
        Args:
            claim_collection: Collection of claims to evaluate
            graph: Optional claim-evidence graph for traceability metrics
            baseline_output: Optional baseline (standard mode) output for comparison
        
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating verifiability metrics")
        
        stats = claim_collection.calculate_statistics()
        
        # Basic metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "session_id": claim_collection.session_id,
            
            # Claim counts
            "total_claims": stats["total_claims"],
            "verified_claims": stats["verified_count"],
            "low_confidence_claims": stats["low_confidence_count"],
            "rejected_claims": stats["rejected_count"],
            
            # Rates
            "rejection_rate": stats["rejection_rate"],
            "verification_rate": stats["verification_rate"],
            "low_confidence_rate": stats["low_confidence_count"] / stats["total_claims"] if stats["total_claims"] > 0 else 0.0,
            
            # Confidence
            "avg_confidence": stats["avg_confidence"],
            
            # Evidence metrics
            "evidence_metrics": self._calculate_evidence_metrics(claim_collection)
        }
        
        # Graph metrics (if available)
        if graph_metrics:
            metrics["graph_metrics"] = self._calculate_graph_metrics(graph_metrics)
        else:
            metrics["graph_metrics"] = None
        
        # Rejection reason breakdown
        metrics["rejection_reasons"] = self.calculate_rejection_reason_breakdown(claim_collection)
        
        # Traceability metrics
        metrics["traceability_metrics"] = self.calculate_traceability_metrics(claim_collection)
        
        # Baseline comparison (if available)
        if baseline_output:
            metrics["baseline_comparison"] = self._compare_to_baseline(
                claim_collection,
                baseline_output
            )
        else:
            # Always include baseline_comparison, even if null
            metrics["baseline_comparison"] = None
        
        # Negative control detection
        negative_control_result = self.detect_negative_control(claim_collection, graph_metrics)
        metrics["negative_control"] = negative_control_result["is_negative_control"]
        metrics["negative_control_details"] = negative_control_result
        
        # Quality flags
        metrics["quality_flags"] = self._generate_quality_flags(metrics)
        
        # Store in history
        self.metrics_history.append(metrics)
        
        logger.info(
            f"Metrics calculated: {metrics['verified_claims']}/{metrics['total_claims']} verified, "
            f"rejection_rate={metrics['rejection_rate']:.2%}"
        )
        
        return metrics
    
    def _calculate_evidence_metrics(
        self,
        collection: ClaimCollection
    ) -> Dict[str, Any]:
        """
        Calculate evidence-related metrics.
        
        Args:
            collection: Claim collection
        
        Returns:
            Evidence metrics dictionary
        """
        if not collection.claims:
            return {
                "avg_evidence_per_claim": 0.0,
                "claims_without_evidence": 0,
                "unsupported_rate": 0.0,
                "avg_evidence_quality": 0.0
            }
        
        # Count evidence
        evidence_counts = [len(c.evidence_ids) for c in collection.claims]
        claims_without_evidence = sum(1 for c in evidence_counts if c == 0)
        
        # Calculate quality (fallback to similarity if no explicit relevance score)
        quality_scores = []
        for claim in collection.claims:
            if claim.evidence_objects:
                scores = []
                for e in claim.evidence_objects:
                    if hasattr(e, "relevance_score"):
                        scores.append(getattr(e, "relevance_score"))
                    else:
                        scores.append(getattr(e, "similarity", 0.0))
                avg_quality = sum(scores) / len(scores) if scores else 0.0
                quality_scores.append(avg_quality)
        
        return {
            "avg_evidence_per_claim": sum(evidence_counts) / len(evidence_counts),
            "min_evidence_per_claim": min(evidence_counts) if evidence_counts else 0,
            "max_evidence_per_claim": max(evidence_counts) if evidence_counts else 0,
            "claims_without_evidence": claims_without_evidence,
            "unsupported_rate": claims_without_evidence / len(collection.claims),
            "avg_evidence_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        }
    
    def _calculate_graph_metrics(
        self,
        graph_metrics: Any
    ) -> Dict[str, Any]:
        """
        Process graph-based traceability metrics.
        
        Args:
            graph_metrics: Pre-computed metrics from ClaimGraph (GraphMetrics object or dict)
        
        Returns:
            Graph metrics dictionary
        """
        if not graph_metrics:
            return {
                "avg_redundancy": 0.0,
                "avg_diversity": 0.0,
                "avg_support_depth": 0.0,
                "conflict_count": 0,
                "total_claims": 0,
                "total_evidence": 0
            }
        
        # Convert GraphMetrics Pydantic object to dict if needed
        if hasattr(graph_metrics, 'model_dump'):
            # Pydantic v2
            metrics_dict = graph_metrics.model_dump()
        elif hasattr(graph_metrics, 'dict'):
            # Pydantic v1
            metrics_dict = graph_metrics.dict()
        elif isinstance(graph_metrics, dict):
            metrics_dict = graph_metrics
        else:
            # Handle as object with attributes (fallback)
            metrics_dict = {
                "avg_redundancy": getattr(graph_metrics, "avg_redundancy", 0.0),
                "avg_diversity": getattr(graph_metrics, "avg_diversity", 0.0),
                "avg_support_depth": getattr(graph_metrics, "avg_support_depth", 0.0),
                "conflict_count": getattr(graph_metrics, "conflict_count", 0),
                "total_claims": getattr(graph_metrics, "total_claims", 0),
                "total_evidence": getattr(graph_metrics, "total_evidence", 0)
            }
        
        return {
            "avg_redundancy": metrics_dict.get("avg_redundancy", 0.0),
            "avg_diversity": metrics_dict.get("avg_diversity", 0.0),
            "avg_support_depth": metrics_dict.get("avg_support_depth", 0.0),
            "conflict_count": metrics_dict.get("conflict_count", 0),
            "total_claims": metrics_dict.get("total_claims", 0),
            "total_evidence": metrics_dict.get("total_evidence", 0)
        }
    
    def _compare_to_baseline(
        self,
        collection: ClaimCollection,
        baseline_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare verifiable mode output to baseline (standard mode).
        
        Args:
            collection: Verifiable mode claim collection
            baseline_output: Standard mode output for comparison
        
        Returns:
            Comparison metrics
        """
        # Count baseline items
        baseline_counts = {
            "concepts": len(baseline_output.get("key_concepts", [])),
            "equations": len(baseline_output.get("equation_explanations", [])),
            "examples": len(baseline_output.get("worked_examples", [])),
            "misconceptions": len(baseline_output.get("common_mistakes", [])),
            "faqs": len(baseline_output.get("faqs", [])),
            "connections": len(baseline_output.get("real_world_connections", []))
        }
        
        baseline_total = sum(baseline_counts.values())
        
        # Verifiable mode counts
        verifiable_total = len(collection.claims)
        verified_count = len(collection.get_verified_claims())
        rejected_count = len(collection.get_rejected_claims())
        
        return {
            "baseline_total_items": baseline_total,
            "verifiable_total_claims": verifiable_total,
            "verifiable_verified_claims": verified_count,
            "verifiable_rejected_claims": rejected_count,
            "reduction_rate": (baseline_total - verified_count) / baseline_total if baseline_total > 0 else 0.0,
            "hallucination_reduction_estimate": rejected_count / baseline_total if baseline_total > 0 else 0.0,
            "baseline_breakdown": baseline_counts
        }
    
    def _generate_quality_flags(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate quality warning flags based on metrics.
        
        Args:
            metrics: Calculated metrics
        
        Returns:
            List of warning flags
        """
        flags = []
        
        # High rejection rate
        if metrics["rejection_rate"] > 0.5:
            flags.append(f"High rejection rate ({metrics['rejection_rate']:.1%})")
        
        # Low verification rate
        if metrics["verification_rate"] < 0.3:
            flags.append(f"Low verification rate ({metrics['verification_rate']:.1%})")
        
        # High unsupported rate
        evidence_metrics = metrics.get("evidence_metrics", {})
        unsupported_rate = evidence_metrics.get("unsupported_rate", 0.0)
        if unsupported_rate > 0.2:
            flags.append(f"High unsupported claim rate ({unsupported_rate:.1%})")
        
        # Low average confidence
        if metrics["avg_confidence"] < 0.4:
            flags.append(f"Low average confidence ({metrics['avg_confidence']:.2f})")
        
        # Poor evidence quality
        avg_evidence_quality = evidence_metrics.get("avg_evidence_quality", 0.0)
        if avg_evidence_quality < 0.5:
            flags.append(f"Low evidence quality ({avg_evidence_quality:.2f})")
        
        return flags
    
    def calculate_rejection_reason_breakdown(
        self,
        collection: ClaimCollection
    ) -> Dict[str, int]:
        """
        Calculate breakdown of rejection reasons.
        
        Args:
            collection: Claim collection
        
        Returns:
            Dict mapping rejection reason to count
        """
        breakdown = {
            "NO_EVIDENCE": 0,
            "LOW_SIMILARITY": 0,
            "INSUFFICIENT_SOURCES": 0,
            "LOW_CONSISTENCY": 0,
            "CONFLICT": 0,
            "INSUFFICIENT_CONFIDENCE": 0,
            "DEPENDENCY_REQUIRED": 0,
            "LOW_CONFIDENCE": 0
        }
        
        for claim in collection.claims:
            if claim.rejection_reason:
                reason_str = claim.rejection_reason.value
                if reason_str in breakdown:
                    breakdown[reason_str] += 1
        
        return breakdown
    
    def calculate_traceability_metrics(
        self,
        collection: ClaimCollection
    ) -> Dict[str, Any]:
        """
        Calculate traceability metrics (evidence linking).
        
        Args:
            collection: Claim collection
        
        Returns:
            Traceability metrics
        """
        if not collection.claims:
            return {
                "traceability_rate": 0.0,
                "claims_with_evidence": 0,
                "claims_without_evidence": 0,
                "avg_evidence_per_verified": 0.0,
                "multi_source_rate": 0.0
            }
        
        claims_with_ev = sum(1 for c in collection.claims if c.evidence_ids)
        claims_without_ev = len(collection.claims) - claims_with_ev
        
        # Multi-source rate
        multi_source = 0
        for claim in collection.claims:
            if len(claim.evidence_ids) >= 2:
                multi_source += 1
        
        # Average evidence for verified claims
        verified_claims = collection.get_verified_claims()
        if verified_claims:
            avg_ev_verified = sum(len(c.evidence_ids) for c in verified_claims) / len(verified_claims)
        else:
            avg_ev_verified = 0.0
        
        return {
            "traceability_rate": claims_with_ev / len(collection.claims),
            "claims_with_evidence": claims_with_ev,
            "claims_without_evidence": claims_without_ev,
            "avg_evidence_per_verified": avg_ev_verified,
            "multi_source_rate": multi_source / len(collection.claims)
        }
    
    def detect_negative_control(
        self,
        collection: ClaimCollection,
        graph_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect negative control scenario (no evidence run).
        
        Negative control is TRUE if ANY of:
        - total_evidence == 0
        - evidence_nodes == 0 (from graph)
        - traceability_rate == 0
        - verified_claims == 0
        
        Args:
            collection: Claim collection
            graph_metrics: Optional graph metrics dict
        
        Returns:
            Negative control info with detailed triggers
        """
        if not collection.claims:
            return {
                "is_negative_control": True,
                "explanation": "No claims extracted (empty session)",
                "trigger_reasons": ["no_claims"]
            }
        
        # Calculate metrics
        total_evidence = sum(len(c.evidence_ids) for c in collection.claims)
        claims_with_evidence = sum(1 for c in collection.claims if c.evidence_ids)
        traceability_rate = claims_with_evidence / len(collection.claims) if collection.claims else 0.0
        verified_count = len(collection.get_verified_claims())
        
        # Check graph metrics for evidence nodes
        evidence_nodes = 0
        if graph_metrics:
            # Handle both dict (backward compat) and GraphMetrics object
            if isinstance(graph_metrics, dict):
                evidence_nodes = graph_metrics.get("evidence_nodes", 0)
            else:
                evidence_nodes = getattr(graph_metrics, "total_evidence", 0)
        
        # Negative control triggers
        is_negative = False
        reasons = []
        
        if total_evidence == 0:
            is_negative = True
            reasons.append("total_evidence==0")
        
        if evidence_nodes == 0:
            is_negative = True
            reasons.append("evidence_nodes==0")
        
        if traceability_rate == 0.0:
            is_negative = True
            reasons.append("traceability_rate==0")
        
        if verified_count == 0:
            is_negative = True
            reasons.append("verified_claims==0")
        
        if is_negative:
            explanation = (
                f"Negative Control Detected: {', '.join(reasons)}. "
                "This means NO EVIDENCE was found in input sources to support AI-generated claims. "
                "High rejection is CORRECT BEHAVIOR - it prevents hallucinations. "
                "Solutions: (1) provide richer input, (2) lower similarity threshold, or (3) accept as valid refusal."
            )
            return {
                "is_negative_control": True,
                "explanation": explanation,
                "trigger_reasons": reasons,
                "total_evidence": total_evidence,
                "evidence_nodes": evidence_nodes,
                "traceability_rate": traceability_rate,
                "verified_count": verified_count
            }
        
        return {
            "is_negative_control": False,
            "explanation": "Normal verification run with evidence present",
            "total_evidence": total_evidence,
            "evidence_nodes": evidence_nodes,
            "traceability_rate": traceability_rate,
            "verified_count": verified_count
        }
    
    def check_input_sufficiency(
        self,
        total_tokens: int,
        chunk_count: int,
        min_tokens: int = 100,
        min_chunks: int = 2
    ) -> Dict[str, Any]:
        """
        Check if input sources are sufficient for reliable verification.
        
        Args:
            total_tokens: Total tokens in source material
            chunk_count: Number of chunks in vector store
            min_tokens: Minimum tokens threshold (default: 100)
            min_chunks: Minimum chunks threshold (default: 2)
        
        Returns:
            Sufficiency check result
        """
        warnings = []
        is_sufficient = True
        
        if total_tokens < min_tokens:
            warnings.append(
                f"⚠️ Limited source material ({total_tokens} tokens). "
                f"Minimum recommended: {min_tokens} tokens. "
                "Verification may be unreliable."
            )
            is_sufficient = False
        
        if chunk_count < min_chunks:
            warnings.append(
                f"⚠️ Few source chunks ({chunk_count}). "
                f"Minimum recommended: {min_chunks} chunks. "
                "Limited diversity for evidence matching."
            )
            is_sufficient = False
        
        return {
            "is_sufficient": is_sufficient,
            "total_tokens": total_tokens,
            "chunk_count": chunk_count,
            "warnings": warnings,
            "recommendation": (
                "Proceed with caution. Consider adding more source material for better coverage."
                if not is_sufficient
                else "Source material is adequate for verification."
            )
        }

    
    def calculate_refusal_metrics(
        self,
        agent_statistics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate refusal metrics from agent statistics.
        
        Args:
            agent_statistics: List of statistics from agents
        
        Returns:
            Refusal metrics dictionary
        """
        if not agent_statistics:
            return {
                "total_agents": 0,
                "total_attempts": 0,
                "total_refusals": 0,
                "avg_refusal_rate": 0.0
            }
        
        total_attempts = sum(s.get("total_attempts", 0) for s in agent_statistics)
        total_refusals = sum(s.get("refused_count", 0) for s in agent_statistics)
        
        return {
            "total_agents": len(agent_statistics),
            "total_attempts": total_attempts,
            "total_refusals": total_refusals,
            "avg_refusal_rate": total_refusals / total_attempts if total_attempts > 0 else 0.0,
            "per_agent": agent_statistics
        }
    
    def generate_report(
        self,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable report from metrics.
        
        Args:
            metrics: Metrics dictionary
        
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "VERIFIABLE MODE METRICS REPORT",
            "=" * 60,
            f"Session: {metrics['session_id']}",
            f"Timestamp: {metrics['timestamp']}",
            "",
            "CLAIM STATISTICS:",
            f"  Total Claims: {metrics['total_claims']}",
            f"  Verified: {metrics['verified_claims']} ({metrics['verification_rate']:.1%})",
            f"  Low Confidence: {metrics['low_confidence_claims']} ({metrics['low_confidence_rate']:.1%})",
            f"  Rejected: {metrics['rejected_claims']} ({metrics['rejection_rate']:.1%})",
            f"  Avg Confidence: {metrics['avg_confidence']:.2f}",
            ""
        ]
        
        # Evidence metrics
        ev_metrics = metrics.get("evidence_metrics", {})
        if ev_metrics:
            lines.extend([
                "EVIDENCE METRICS:",
                f"  Avg Evidence/Claim: {ev_metrics.get('avg_evidence_per_claim', 0):.1f}",
                f"  Unsupported Claims: {ev_metrics.get('claims_without_evidence', 0)}",
                f"  Unsupported Rate: {ev_metrics.get('unsupported_rate', 0):.1%}",
                f"  Avg Evidence Quality: {ev_metrics.get('avg_evidence_quality', 0):.2f}",
                ""
            ])
        
        # Baseline comparison
        if "baseline_comparison" in metrics:
            comp = metrics["baseline_comparison"]
            lines.extend([
                "BASELINE COMPARISON:",
                f"  Baseline Items: {comp['baseline_total_items']}",
                f"  Verifiable Verified: {comp['verifiable_verified_claims']}",
                f"  Reduction Rate: {comp['reduction_rate']:.1%}",
                f"  Est. Hallucination Reduction: {comp['hallucination_reduction_estimate']:.1%}",
                ""
            ])
        
        # Quality flags
        if metrics.get("quality_flags"):
            lines.extend([
                "QUALITY FLAGS:",
                *[f"  ⚠️ {flag}" for flag in metrics["quality_flags"]],
                ""
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def export_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Export full metrics history.
        
        Returns:
            List of all calculated metrics
        """
        return self.metrics_history
