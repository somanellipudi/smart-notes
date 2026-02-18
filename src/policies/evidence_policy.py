"""
Evidence Sufficiency Policy - Deterministic Decision Rules

This module implements explicit, reproducible rules for determining whether
evidence is sufficient to verify a claim.

Policy:
- Use adaptive entailment threshold based on session distribution
- Require min_supporting_sources (default 2 independent sources)
- Allow single-source systems to pass with 1 source
- Require max_contradiction_prob (default 0.35)
- Define "independent source" = different source_id or source_type

Reference: Selena's research-rigor requirements for evidence sufficiency.
"""

import logging
import math
from typing import List, Optional
from dataclasses import dataclass

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
    required_sources: int
    entailment_threshold_used: float
    max_entailment: float
    max_contradiction: float


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


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * (percentile / 100.0)
    lower_index = int(math.floor(k))
    upper_index = int(math.ceil(k))
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * (k - lower_index)


def evaluate_evidence_sufficiency(
    claim: LearningClaim,
    evidence_items: List[EvidenceItem],
    nli_results: Optional[List[dict]] = None,
    entailment_score_distribution: Optional[List[float]] = None,
    min_entailment_prob: float = None,
    min_supporting_sources: int = None,
    max_contradiction_prob: float = None,
    total_independent_sources: Optional[int] = None
) -> SufficiencyDecision:
    """
    Evaluate whether evidence is sufficient to verify a claim.
    
    Deterministic decision rules:
    1. If no evidence, return REJECTED (NO_EVIDENCE)
    2. If max_entailment < adaptive_threshold, return REJECTED (INSUFFICIENT_CONFIDENCE)
    3. If independent_sources < required_sources, return LOW_CONFIDENCE (INSUFFICIENT_SOURCES)
    4. If max_contradiction > max_contradiction_prob, return LOW_CONFIDENCE (CONFLICT)
    5. Otherwise, return VERIFIED
    
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
        entailment_score_distribution: Optional list of entailment scores across claims
        min_entailment_prob: Minimum entailment probability (default from config)
        min_supporting_sources: Minimum independent sources (default from config)
        max_contradiction_prob: Maximum contradiction probability (default from config)
        total_independent_sources: Total independent sources available in the system
    
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
        max_contradiction_prob = 0.35

    required_sources = min_supporting_sources
    if total_independent_sources is not None and total_independent_sources <= 1:
        required_sources = 1
    
    # Rule 1: No evidence → REJECTED
    if not evidence_items:
        logger.debug(f"Claim {claim.claim_id[:8]}: No evidence → REJECTED")
        return SufficiencyDecision(
            status_override=VerificationStatus.REJECTED,
            reason="No evidence sources found",
            support_count=0,
            contradiction_count=0,
            confidence_score=0.0,
            independent_sources=0,
            required_sources=required_sources,
            entailment_threshold_used=min_entailment_prob,
            max_entailment=0.0,
            max_contradiction=0.0
        )
    
    support_count = len(evidence_items)
    independent_sources = count_independent_sources(evidence_items)
    
    # Analyze NLI results if provided
    max_entailment = 0.0
    max_contradiction = 0.0
    if nli_results:
        # Aggregate NLI scores
        entailment_scores = [r["score"] for r in nli_results if r.get("label") == "entailment"]
        contradiction_scores = [r["score"] for r in nli_results if r.get("label") == "contradiction"]

        max_entailment = max(entailment_scores) if entailment_scores else 0.0
        max_contradiction = max(contradiction_scores) if contradiction_scores else 0.0
    else:
        # If no NLI results, use similarity scores as proxy
        # High similarity → high entailment probability
        if evidence_items:
            similarities = [ev.similarity for ev in evidence_items if ev.similarity is not None]
            max_entailment = max(similarities) if similarities else 0.0
            max_contradiction = 0.0  # No contradiction detection without NLI
    
    contradiction_count = len([r for r in (nli_results or []) if r.get("label") == "contradiction"])
    
    adaptive_threshold = min_entailment_prob
    if entailment_score_distribution:
        adaptive_threshold = max(0.50, _percentile(entailment_score_distribution, 40))

    # Rule 2: Low entailment probability → REJECTED
    if max_entailment < adaptive_threshold:
        logger.debug(
            f"Claim {claim.claim_id[:8]}: "
            f"max_entailment={max_entailment:.2f} < {adaptive_threshold:.2f} → REJECTED"
        )
        return SufficiencyDecision(
            status_override=VerificationStatus.REJECTED,
            reason=f"Insufficient entailment probability ({max_entailment:.2f} < {adaptive_threshold:.2f})",
            support_count=support_count,
            contradiction_count=contradiction_count,
            confidence_score=max_entailment,
            independent_sources=independent_sources,
            required_sources=required_sources,
            entailment_threshold_used=adaptive_threshold,
            max_entailment=max_entailment,
            max_contradiction=max_contradiction
        )
    
    # Rule 3: Insufficient independent sources → LOW_CONFIDENCE
    if independent_sources < required_sources:
        logger.debug(
            f"Claim {claim.claim_id[:8]}: "
            f"independent_sources={independent_sources} < {required_sources} → LOW_CONFIDENCE"
        )
        return SufficiencyDecision(
            status_override=VerificationStatus.LOW_CONFIDENCE,
            reason=f"Insufficient independent sources ({independent_sources} < {required_sources})",
            support_count=support_count,
            contradiction_count=contradiction_count,
            confidence_score=max_entailment * 0.7,
            independent_sources=independent_sources,
            required_sources=required_sources,
            entailment_threshold_used=adaptive_threshold,
            max_entailment=max_entailment,
            max_contradiction=max_contradiction
        )
    
    # Rule 4: High contradiction probability → LOW_CONFIDENCE
    if max_contradiction > max_contradiction_prob:
        logger.debug(
            f"Claim {claim.claim_id[:8]}: "
            f"max_contradiction={max_contradiction:.2f} > {max_contradiction_prob:.2f} → LOW_CONFIDENCE"
        )
        return SufficiencyDecision(
            status_override=VerificationStatus.LOW_CONFIDENCE,
            reason=f"High contradiction probability ({max_contradiction:.2f} > {max_contradiction_prob:.2f})",
            support_count=support_count,
            contradiction_count=contradiction_count,
            confidence_score=max_entailment * 0.5,
            independent_sources=independent_sources,
            required_sources=required_sources,
            entailment_threshold_used=adaptive_threshold,
            max_entailment=max_entailment,
            max_contradiction=max_contradiction
        )
    
    # Rule 6: All checks passed → VERIFIED
    logger.debug(
        f"Claim {claim.claim_id[:8]}: "
        f"All sufficiency checks passed → VERIFIED "
        f"(entailment={max_entailment:.2f}, sources={independent_sources})"
    )
    return SufficiencyDecision(
        status_override=VerificationStatus.VERIFIED,
        reason=f"Sufficient evidence with {independent_sources} independent sources",
        support_count=support_count,
        contradiction_count=contradiction_count,
        confidence_score=max_entailment,
        independent_sources=independent_sources,
        required_sources=required_sources,
        entailment_threshold_used=adaptive_threshold,
        max_entailment=max_entailment,
        max_contradiction=max_contradiction
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
        if decision.independent_sources < decision.required_sources:
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
        "required_sources": decision.required_sources,
        "entailment_threshold_used": decision.entailment_threshold_used,
        "max_entailment": decision.max_entailment,
        "max_contradiction": decision.max_contradiction,
        "applied_at": "evidence_policy"
    }
    
    logger.info(
        f"Applied sufficiency policy to claim {claim.claim_id[:8]}: "
        f"{decision.status_override} (confidence={decision.confidence_score:.2f})"
    )
