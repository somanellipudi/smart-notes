"""
Deterministic verification policy for domain-scoped claim validation.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.claims.schema import (
    LearningClaim,
    EvidenceItem,
    VerificationStatus,
    RejectionReason,
    ClaimType
)
from src.policies.domain_profiles import DomainProfile
from src.policies.evidence_policy import count_independent_sources
import config


BIG_O_PATTERN = re.compile(r"\b[OΘΩ]\s*\(\s*[^)]+\)")
PSEUDOCODE_TOKENS = re.compile(
    r"\b(for|while|if|else|return|break|continue|swap|push|pop|enqueue|dequeue)\b|"
    r"<-|:=|\bdo\b|\bend\b|\bthen\b"
)
INVARIANT_TOKENS = re.compile(r"\b(invariant|maintains|preserves)\b", re.IGNORECASE)


@dataclass
class VerificationDecision:
    status: VerificationStatus
    rejection_reason: Optional[RejectionReason]
    confidence: float


def _extract_text_for_tagging(claim: LearningClaim) -> str:
    return (
        claim.claim_text
        or claim.metadata.get("draft_text", "")
        or claim.metadata.get("ui_display", "")
        or ""
    )


def _detect_claim_type(text: str) -> Optional[ClaimType]:
    text_lower = text.lower()

    if BIG_O_PATTERN.search(text):
        return ClaimType.COMPLEXITY

    if INVARIANT_TOKENS.search(text):
        return ClaimType.INVARIANT

    if PSEUDOCODE_TOKENS.search(text_lower):
        return ClaimType.ALGORITHM_STEP

    return None


def tag_claim_type(claim: LearningClaim, domain_profile: DomainProfile) -> None:
    """Tag claim type using deterministic heuristics for CS domains."""
    if domain_profile.name not in {"algorithms", "cs"}:
        return

    text = _extract_text_for_tagging(claim)
    if not text:
        return

    detected = _detect_claim_type(text)
    if detected is None:
        return

    if detected.value in domain_profile.allowed_claim_types:
        claim.claim_type = detected


def _evidence_contains_bigo(evidence_items: List[EvidenceItem]) -> bool:
    for ev in evidence_items:
        if BIG_O_PATTERN.search(ev.snippet):
            return True
    return False


def _evidence_contains_pseudocode(evidence_items: List[EvidenceItem]) -> bool:
    for ev in evidence_items:
        if PSEUDOCODE_TOKENS.search(ev.snippet.lower()):
            return True
    return False


def _evidence_contains_invariant(evidence_items: List[EvidenceItem]) -> bool:
    for ev in evidence_items:
        if INVARIANT_TOKENS.search(ev.snippet):
            return True
    return False


def evaluate_claim(
    claim: LearningClaim,
    evidence_items: List[EvidenceItem],
    domain_profile: DomainProfile,
    nli_results: Optional[List[dict]] = None
) -> VerificationDecision:
    """
    Deterministic decision policy based on evidence and domain rules.
    """
    allowed_types = {ct for ct in domain_profile.allowed_claim_types}
    if claim.claim_type.value not in allowed_types:
        return VerificationDecision(
            status=VerificationStatus.REJECTED,
            rejection_reason=RejectionReason.DISALLOWED_CLAIM_TYPE,
            confidence=0.0
        )

    if not evidence_items:
        return VerificationDecision(
            status=VerificationStatus.REJECTED,
            rejection_reason=RejectionReason.NO_EVIDENCE,
            confidence=0.0
        )

    similarities = [ev.similarity for ev in evidence_items if ev.similarity is not None]
    max_similarity = max(similarities) if similarities else 0.0

    if nli_results:
        entailment_scores = [r.get("score", 0.0) for r in nli_results if r.get("label") == "entailment"]
        confidence = max(entailment_scores) if entailment_scores else max_similarity
    else:
        confidence = max_similarity

    if max_similarity < config.VERIFIABLE_RELEVANCE_THRESHOLD:
        return VerificationDecision(
            status=VerificationStatus.REJECTED,
            rejection_reason=RejectionReason.LOW_SIMILARITY,
            confidence=confidence
        )

    if claim.claim_type == ClaimType.COMPLEXITY:
        if not _evidence_contains_bigo(evidence_items):
            return VerificationDecision(
                status=VerificationStatus.REJECTED,
                rejection_reason=RejectionReason.MISSING_BIGO,
                confidence=confidence
            )

    if claim.claim_type == ClaimType.ALGORITHM_STEP:
        if not _evidence_contains_pseudocode(evidence_items):
            return VerificationDecision(
                status=VerificationStatus.REJECTED,
                rejection_reason=RejectionReason.MISSING_PSEUDOCODE,
                confidence=confidence
            )

    if claim.claim_type == ClaimType.INVARIANT:
        if not _evidence_contains_invariant(evidence_items):
            return VerificationDecision(
                status=VerificationStatus.LOW_CONFIDENCE,
                rejection_reason=RejectionReason.LOW_CONFIDENCE_GENERIC,
                confidence=confidence
            )

    independent_sources = count_independent_sources(evidence_items)
    if domain_profile.name in {"algorithms", "cs"}:
        required_sources = 1
    else:
        required_sources = max(1, config.MIN_SUPPORTING_SOURCES)

    if independent_sources < required_sources:
        return VerificationDecision(
            status=VerificationStatus.LOW_CONFIDENCE,
            rejection_reason=RejectionReason.INSUFFICIENT_SOURCES,
            confidence=confidence
        )

    if confidence >= config.VERIFIABLE_VERIFIED_THRESHOLD:
        return VerificationDecision(
            status=VerificationStatus.VERIFIED,
            rejection_reason=None,
            confidence=confidence
        )

    if confidence < config.VERIFIABLE_REJECTED_THRESHOLD:
        return VerificationDecision(
            status=VerificationStatus.REJECTED,
            rejection_reason=RejectionReason.INSUFFICIENT_CONFIDENCE,
            confidence=confidence
        )

    return VerificationDecision(
        status=VerificationStatus.LOW_CONFIDENCE,
        rejection_reason=RejectionReason.LOW_CONFIDENCE_GENERIC,
        confidence=confidence
    )
