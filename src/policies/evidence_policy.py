"""
Evidence Sufficiency Policy - Deterministic Decision Rules

This module implements explicit, reproducible rules for determining whether
evidence is sufficient to verify a claim.

Policy:
- Require min_entailment_prob (default 0.60)
- Require min_supporting_sources (default 2 independent sources)
- Require max_contradiction_prob (default 0.30)
- Define "independent source" = different source_id or source_type

Reference: Selena's research-rigor requirements for evidence sufficiency.
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from src.claims.schema import (
    LearningClaim,
    EvidenceItem,
    VerificationStatus,
    RejectionReason
)
import config

logger = logging.getLogger(__name__)


@dataclass
class SufficiencyDecision:
    """
    Result of evidence sufficiency evaluation.
    
    Attributes:
        status_override: Suggested verification status (VERIFIED/LOW_CONFIDENCE/REJECTED)
        reason: Human-readable explanation
        support_count: Number of supporting evidence items
        contradiction_count: Number of contradicting evidence items (if available)
        confidence_score: Computed confidence score (0-1)
        independent_sources: Number of independent sources found
    """
    status_override: VerificationStatus
    reason: str
    support_count: int
    contradiction_count: int
    confidence_score: float
    independent_sources: int


def count_independent_sources(evidence_items: List[EvidenceItem]) -> int:
    """
    Count independent sources in evidence list.
    
    Two evidence items are from independent sources if they have:
    - Different source_id, OR
    - Different source_type (if source_id is missing/same)
    
    Args:
        evidence_items: List of evidence items
    
    Returns:
        Number of unique independent sources
    
    Examples:
        >>> ev1 = EvidenceItem(source_id="lec1", source_type="transcript", snippet="...")
        >>> ev2 = EvidenceItem(source_id="lec2", source_type="transcript", snippet="...")
        >>> count_independent_sources([ev1, ev2])
        2
        >>> ev3 = EvidenceItem(source_id="lec1", source_type="notes", snippet="...")
        >>> count_independent_sources([ev1, ev3])
        2
    """
    if not evidence_items:
        return 0
    
    # Track unique (source_id, source_type) tuples
    unique_sources = set()
    for ev in evidence_items:
        # Use tuple of (source_id, source_type) to identify unique sources
        unique_sources.add((ev.source_id, ev.source_type))
    
    return len(unique_sources)


def evaluate_evidence_sufficiency(
    claim: LearningClaim,
    evidence_items: List[EvidenceItem],
    nli_results: Optional[List[dict]] = None,
    min_entailment_prob: float = None,
    min_supporting_sources: int = None,
    max_contradiction_prob: float = None
) -> SufficiencyDecision:
    """
    Evaluate whether evidence is sufficient to verify a claim.
    
    Deterministic decision rules:
    1. If no evidence, return REJECTED (NO_EVIDENCE)
    2. If entailment_prob < min_entailment_prob, return REJECTED (INSUFFICIENT_CONFIDENCE)
    3. If independent_sources < min_supporting_sources, return LOW_CONFIDENCE (INSUFFICIENT_SOURCES)
    4. If contradiction_prob > max_contradiction_prob, return LOW_CONFIDENCE (CONFLICT)
    5. If both entailment and contradiction exist, return LOW_CONFIDENCE (CONFLICT)
    6. Otherwise, return VERIFIED
    
    Relaxed Mode:
    If config.RELAXED_VERIFICATION_MODE is True, uses relaxed thresholds:
    - MIN_ENTAILMENT_PROB = 0.50
    - MIN_SUPPORTING_SOURCES = 1
    - MAX_CONTRADICTION_PROB = 0.50
    
    Args:
        claim: Learning claim to evaluate
        evidence_items: List of evidence supporting the claim
        nli_results: Optional NLI results with entailment/contradiction probabilities
                     Format: [{"label": "entailment/contradiction/neutral", "score": float}, ...]
        min_entailment_prob: Minimum entailment probability (default from config)
        min_supporting_sources: Minimum independent sources (default from config)
        max_contradiction_prob: Maximum contradiction probability (default from config)
    
    Returns:
        SufficiencyDecision with status override and explanation
    
    Examples:
        >>> claim = LearningClaim(claim_type="definition", claim_text="F=ma")
        >>> ev1 = EvidenceItem(source_id="lec1", source_type="transcript", snippet="Force equals mass times acceleration")
        >>> ev2 = EvidenceItem(source_id="notes", source_type="notes", snippet="Newton's second law: F=ma")
        >>> decision = evaluate_evidence_sufficiency(claim, [ev1, ev2])
        >>> decision.status_override
        VerificationStatus.VERIFIED
    """
    # Apply relaxed mode if enabled
    if config.RELAXED_VERIFICATION_MODE:
        if min_entailment_prob is None:
            min_entailment_prob = config.RELAXED_MIN_ENTAILMENT_PROB
        if min_supporting_sources is None:
            min_supporting_sources = config.RELAXED_MIN_SUPPORTING_SOURCES
        if max_contradiction_prob is None:
            max_contradiction_prob = config.RELAXED_MAX_CONTRADICTION_PROB
        logger.info("Running in RELAXED_VERIFICATION_MODE")
    
    # Use config defaults if not provided
    if min_entailment_prob is None:
        min_entailment_prob = config.MIN_ENTAILMENT_PROB
    if min_supporting_sources is None:
        min_supporting_sources = config.MIN_SUPPORTING_SOURCES
    if max_contradiction_prob is None:
        max_contradiction_prob = config.MAX_CONTRADICTION_PROB
    
    # Rule 1: No evidence → REJECTED
    if not evidence_items:
        logger.debug(f"Claim {claim.claim_id[:8]}: No evidence → REJECTED")
        return SufficiencyDecision(
            status_override=VerificationStatus.REJECTED,
            reason="No evidence sources found",
            support_count=0,
            contradiction_count=0,
            confidence_score=0.0,
            independent_sources=0
        )
    
    support_count = len(evidence_items)
    independent_sources = count_independent_sources(evidence_items)
    
    # Analyze NLI results if provided
    entailment_prob = 0.0
    contradiction_prob = 0.0
    if nli_results:
        # Aggregate NLI scores
        entailment_scores = [r["score"] for r in nli_results if r.get("label") == "entailment"]
        contradiction_scores = [r["score"] for r in nli_results if r.get("label") == "contradiction"]
        
        entailment_prob = max(entailment_scores) if entailment_scores else 0.0
        contradiction_prob = max(contradiction_scores) if contradiction_scores else 0.0
    else:
        # If no NLI results, use similarity scores as proxy
        # High similarity → high entailment probability
        if evidence_items:
            avg_similarity = sum(ev.similarity for ev in evidence_items) / len(evidence_items)
            entailment_prob = avg_similarity
            contradiction_prob = 0.0  # No contradiction detection without NLI
    
    contradiction_count = len([r for r in (nli_results or []) if r.get("label") == "contradiction"])
    
    # Rule 2: Low entailment probability → REJECTED
    if entailment_prob < min_entailment_prob:
        logger.debug(
            f"Claim {claim.claim_id[:8]}: "
            f"entailment_prob={entailment_prob:.2f} < {min_entailment_prob} → REJECTED"
        )
        return SufficiencyDecision(
            status_override=VerificationStatus.REJECTED,
            reason=f"Insufficient entailment probability ({entailment_prob:.2f} < {min_entailment_prob})",
            support_count=support_count,
            contradiction_count=contradiction_count,
            confidence_score=entailment_prob,
            independent_sources=independent_sources
        )
    
    # Rule 3: Insufficient independent sources → LOW_CONFIDENCE
    if independent_sources < min_supporting_sources:
        logger.debug(
            f"Claim {claim.claim_id[:8]}: "
            f"independent_sources={independent_sources} < {min_supporting_sources} → LOW_CONFIDENCE"
        )
        return SufficiencyDecision(
            status_override=VerificationStatus.LOW_CONFIDENCE,
            reason=f"Insufficient independent sources ({independent_sources} < {min_supporting_sources})",
            support_count=support_count,
            contradiction_count=contradiction_count,
            confidence_score=entailment_prob * 0.7,  # Penalize for insufficient sources
            independent_sources=independent_sources
        )
    
    # Rule 4: High contradiction probability → LOW_CONFIDENCE
    if contradiction_prob > max_contradiction_prob:
        logger.debug(
            f"Claim {claim.claim_id[:8]}: "
            f"contradiction_prob={contradiction_prob:.2f} > {max_contradiction_prob} → LOW_CONFIDENCE"
        )
        return SufficiencyDecision(
            status_override=VerificationStatus.LOW_CONFIDENCE,
            reason=f"High contradiction probability ({contradiction_prob:.2f} > {max_contradiction_prob})",
            support_count=support_count,
            contradiction_count=contradiction_count,
            confidence_score=entailment_prob * 0.5,  # Heavily penalize for contradiction
            independent_sources=independent_sources
        )
    
    # Rule 5: Both entailment and contradiction exist (with significant scores) → LOW_CONFIDENCE (conflict)
    if entailment_prob >= min_entailment_prob and contradiction_prob > max_contradiction_prob:
        logger.debug(
            f"Claim {claim.claim_id[:8]}: "
            f"Both strong entailment and contradiction detected → LOW_CONFIDENCE"
        )
        return SufficiencyDecision(
            status_override=VerificationStatus.LOW_CONFIDENCE,
            reason=f"Conflicting evidence detected (entailment={entailment_prob:.2f}, contradiction={contradiction_prob:.2f})",
            support_count=support_count,
            contradiction_count=contradiction_count,
            confidence_score=entailment_prob * 0.6,  # Penalize for conflict
            independent_sources=independent_sources
        )
    
    # Rule 6: All checks passed → VERIFIED
    logger.debug(
        f"Claim {claim.claim_id[:8]}: "
        f"All sufficiency checks passed → VERIFIED "
        f"(entailment={entailment_prob:.2f}, sources={independent_sources})"
    )
    return SufficiencyDecision(
        status_override=VerificationStatus.VERIFIED,
        reason=f"Sufficient evidence with {independent_sources} independent sources",
        support_count=support_count,
        contradiction_count=contradiction_count,
        confidence_score=entailment_prob,
        independent_sources=independent_sources
    )


def apply_sufficiency_policy(
    claim: LearningClaim,
    evidence_items: List[EvidenceItem],
    nli_results: Optional[List[dict]] = None
) -> None:
    """
    Apply evidence sufficiency policy to a claim (in-place update).
    
    Updates claim.status, claim.confidence, and claim.rejection_reason
    based on evidence sufficiency evaluation.
    
    Args:
        claim: Learning claim to update
        evidence_items: Evidence supporting the claim
        nli_results: Optional NLI results
    
    Side effects:
        - Updates claim.status
        - Updates claim.confidence
        - Updates claim.rejection_reason if applicable
        - Adds sufficiency metadata to claim.metadata
    """
    decision = evaluate_evidence_sufficiency(claim, evidence_items, nli_results)
    
    # Update claim based on decision
    claim.status = decision.status_override
    claim.confidence = decision.confidence_score
    
    # Set rejection reason if not verified
    if decision.status_override == VerificationStatus.REJECTED:
        if decision.support_count == 0:
            claim.rejection_reason = RejectionReason.NO_EVIDENCE
        else:
            claim.rejection_reason = RejectionReason.INSUFFICIENT_CONFIDENCE
    elif decision.status_override == VerificationStatus.LOW_CONFIDENCE:
        if decision.independent_sources < config.MIN_SUPPORTING_SOURCES:
            claim.rejection_reason = RejectionReason.INSUFFICIENT_SOURCES
        elif decision.contradiction_count > 0:
            claim.rejection_reason = RejectionReason.CONFLICT
        else:
            claim.rejection_reason = RejectionReason.LOW_CONFIDENCE_GENERIC
    else:
        claim.rejection_reason = None
    
    # Add sufficiency metadata
    claim.metadata["sufficiency_decision"] = {
        "reason": decision.reason,
        "support_count": decision.support_count,
        "contradiction_count": decision.contradiction_count,
        "independent_sources": decision.independent_sources,
        "applied_at": "evidence_policy"
    }
    
    logger.info(
        f"Applied sufficiency policy to claim {claim.claim_id[:8]}: "
        f"{decision.status_override} (confidence={decision.confidence_score:.2f})"
    )
