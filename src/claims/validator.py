"""
Claim validator for promoting, rejecting, and filtering claims.

Implements validation rules to determine which claims should be
verified, kept as low-confidence, or rejected entirely.
"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime

from src.claims.schema import (
    LearningClaim,
    ClaimCollection,
    VerificationStatus,
    RejectionReason,
    ClaimType
)

logger = logging.getLogger(__name__)


class ClaimValidator:
    """
    Validates claims and updates their status based on evidence and confidence.
    
    Implements failure-aware validation that explicitly rejects claims
    when evidence is insufficient rather than generating them anyway.
    """
    
    def __init__(
        self,
        verified_threshold: float = 0.7,
        rejected_threshold: float = 0.3,
        min_evidence_count: int = 1,
        strict_mode: bool = True
    ):
        """
        Initialize claim validator.
        
        Args:
            verified_threshold: Confidence threshold for VERIFIED status
            rejected_threshold: Confidence below this is REJECTED
            min_evidence_count: Minimum evidence items required
            strict_mode: If True, apply stricter validation rules
        """
        self.verified_threshold = verified_threshold
        self.rejected_threshold = rejected_threshold
        self.min_evidence_count = min_evidence_count
        self.strict_mode = strict_mode
        
        self.validation_count = 0
        self.promoted_count = 0
        self.rejected_count = 0
        
        logger.info(
            f"ClaimValidator initialized: "
            f"verified>={verified_threshold}, "
            f"rejected<{rejected_threshold}, "
            f"min_evidence={min_evidence_count}, "
            f"strict={strict_mode}"
        )
    
    def validate_claim(self, claim: LearningClaim) -> VerificationStatus:
        """
        Validate a single claim and determine its status.
        
        Handles different claim types:
        - QUESTION: Skips verification, requires answer generation
        - MISCONCEPTION: Needs proper framing check
        - FACT_CLAIM and others: Standard verification
        
        Args:
            claim: LearningClaim to validate
        
        Returns:
            New VerificationStatus
        """
        self.validation_count += 1
        
        # Route based on claim type
        if claim.claim_type == ClaimType.QUESTION:
            logger.debug(f"Skipping verification for QUESTION: '{claim.claim_text[:50]}...'")
            # Questions should be handled by QuestionAnswerer, not standard verification
            # If we reach here without answer, mark as needs processing
            if not claim.answer_text:
                claim.status = VerificationStatus.REJECTED
                claim.rejection_reason = RejectionReason.DISALLOWED_CLAIM_TYPE
                return VerificationStatus.REJECTED
            else:
                # Already answered
                return VerificationStatus.ANSWERED_WITH_CITATIONS
        
        if claim.claim_type == ClaimType.MISCONCEPTION:
            logger.debug(f"Processing MISCONCEPTION: '{claim.claim_text[:50]}...'")
            # Check if misconception is properly framed with correction
            if not self._has_proper_framing(claim):
                claim.status = VerificationStatus.NEEDS_FRAMING
                claim.rejection_reason = None
                return VerificationStatus.NEEDS_FRAMING
            # If framed properly, verify the correction statement
            # Fall through to standard verification
        
        # Standard verification for FACT_CLAIM and other types
        # Check evidence count
        has_sufficient_evidence = len(claim.evidence_ids) >= self.min_evidence_count
        has_conflicts = bool(claim.metadata.get("evidence_conflicts"))
        consistency_score = float(claim.metadata.get("consistency_score", 1.0))
        evidence_sufficient = bool(claim.metadata.get("evidence_sufficient", has_sufficient_evidence))
        
        if not evidence_sufficient:
            logger.debug(
                f"Rejecting claim (insufficient evidence): "
                f"{claim.claim_text[:50]}..."
            )
            claim.rejection_reason = RejectionReason.NO_EVIDENCE
            self.rejected_count += 1
            return VerificationStatus.REJECTED
        
        if has_conflicts:
            logger.debug(
                f"Rejecting claim (conflicting evidence): "
                f"{claim.claim_text[:50]}..."
            )
            claim.rejection_reason = RejectionReason.CONFLICT
            self.rejected_count += 1
            return VerificationStatus.REJECTED
        
        if consistency_score < self.rejected_threshold:
            logger.debug(
                f"Rejecting claim (low consistency {consistency_score:.2f}): "
                f"{claim.claim_text[:50]}..."
            )
            claim.rejection_reason = RejectionReason.LOW_CONSISTENCY
            self.rejected_count += 1
            return VerificationStatus.REJECTED
        
        # Check confidence
        if claim.confidence >= self.verified_threshold and claim.claim_text.strip():
            logger.debug(
                f"Verifying claim (high confidence {claim.confidence:.2f}): "
                f"{claim.claim_text[:50]}..."
            )
            claim.rejection_reason = None
            self.promoted_count += 1
            return VerificationStatus.VERIFIED
        
        elif claim.confidence < self.rejected_threshold:
            logger.debug(
                f"Rejecting claim (low confidence {claim.confidence:.2f}): "
                f"{claim.claim_text[:50]}..."
            )
            claim.rejection_reason = RejectionReason.INSUFFICIENT_CONFIDENCE
            self.rejected_count += 1
            return VerificationStatus.REJECTED
        
        else:
            # Medium confidence - keep as LOW_CONFIDENCE
            logger.debug(
                f"Low confidence claim (conf {claim.confidence:.2f}): "
                f"{claim.claim_text[:50]}..."
            )
            return VerificationStatus.LOW_CONFIDENCE
    
    def _has_proper_framing(self, claim: LearningClaim) -> bool:
        """
        Check if misconception has proper framing with correction.
        
        Args:
            claim: LearningClaim with type=MISCONCEPTION
        
        Returns:
            True if misconception is properly framed
        """
        text = claim.claim_text.lower()
        
        # Check for correction indicators (NOT including "misconception" itself)
        correction_indicators = [
            "actually", "in reality", "correct",
            "instead", "rather", "false",
            "wrong", "incorrect"
        ]
        
        has_correction = any(ind in text for ind in correction_indicators)
        
        # Check metadata for explicit framing
        has_framing_metadata = (
            claim.metadata.get("has_correction") or
            claim.metadata.get("correction_statement")
        )
        
        return has_correction or has_framing_metadata
    
    def validate_collection(
        self,
        collection: ClaimCollection,
        update_in_place: bool = True
    ) -> ClaimCollection:
        """
        Validate all claims in a collection.
        
        Args:
            collection: ClaimCollection to validate
            update_in_place: If True, update claims in original collection
        
        Returns:
            Updated ClaimCollection
        """
        logger.info(f"Validating {len(collection.claims)} claims")
        
        for claim in collection.claims:
            # Validate and update status
            new_status = self.validate_claim(claim)
            
            if update_in_place:
                claim.status = new_status
                claim.validated_at = datetime.now()
        
        logger.info(
            f"Validation complete: "
            f"{self.promoted_count} verified, "
            f"{self.rejected_count} rejected, "
            f"{self.validation_count - self.promoted_count - self.rejected_count} low-confidence"
        )
        
        return collection
    
    def filter_collection(
        self,
        collection: ClaimCollection,
        include_verified: bool = True,
        include_low_confidence: bool = False,
        include_rejected: bool = False
    ) -> ClaimCollection:
        """
        Filter collection by claim status.
        
        Args:
            collection: ClaimCollection to filter
            include_verified: Include VERIFIED claims
            include_low_confidence: Include LOW_CONFIDENCE claims
            include_rejected: Include REJECTED claims
        
        Returns:
            New ClaimCollection with filtered claims
        """
        filtered_claims = []
        
        for claim in collection.claims:
            if claim.status == VerificationStatus.VERIFIED and include_verified:
                filtered_claims.append(claim)
            elif claim.status == VerificationStatus.LOW_CONFIDENCE and include_low_confidence:
                filtered_claims.append(claim)
            elif claim.status == VerificationStatus.REJECTED and include_rejected:
                filtered_claims.append(claim)
        
        filtered_collection = ClaimCollection(
            session_id=collection.session_id,
            claims=filtered_claims,
            metadata={
                **collection.metadata,
                "filtered": True,
                "filter_config": {
                    "include_verified": include_verified,
                    "include_low_confidence": include_low_confidence,
                    "include_rejected": include_rejected
                }
            }
        )
        
        logger.info(
            f"Filtered {len(collection.claims)} claims to {len(filtered_claims)}"
        )
        
        return filtered_collection
    
    def get_rejected_claims_report(
        self,
        collection: ClaimCollection
    ) -> Dict[str, Any]:
        """
        Generate a report on rejected claims.
        
        Args:
            collection: ClaimCollection to analyze
        
        Returns:
            Dictionary with rejection analysis
        """
        rejected_claims = collection.get_rejected_claims()
        
        if not rejected_claims:
            return {
                "rejected_count": 0,
                "rejection_reasons": {},
                "examples": []
            }
        
        # Analyze rejection reasons
        reasons = {
            "no_evidence": 0,
            "low_confidence": 0,
            "insufficient_evidence": 0
        }
        
        examples = []
        
        for claim in rejected_claims[:5]:  # Limit examples
            # Determine reason
            if len(claim.evidence_ids) == 0:
                reason = "no_evidence"
                reasons["no_evidence"] += 1
            elif len(claim.evidence_ids) < self.min_evidence_count:
                reason = "insufficient_evidence"
                reasons["insufficient_evidence"] += 1
            else:
                reason = "low_confidence"
                reasons["low_confidence"] += 1
            
            examples.append({
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text[:100],
                "claim_type": claim.claim_type.value,
                "confidence": claim.confidence,
                "evidence_count": len(claim.evidence_ids),
                "reason": reason
            })
        
        return {
            "rejected_count": len(rejected_claims),
            "rejection_rate": len(rejected_claims) / len(collection.claims) if collection.claims else 0.0,
            "rejection_reasons": reasons,
            "examples": examples
        }
    
    def apply_type_specific_rules(
        self,
        claim: LearningClaim
    ) -> bool:
        """
        Apply claim-type-specific validation rules.
        
        Args:
            claim: Claim to validate
        
        Returns:
            True if claim passes type-specific validation
        """
        if not self.strict_mode:
            return True
        
        # Definition claims need clear definitional evidence
        if claim.claim_type == ClaimType.DEFINITION:
            # Check that claim text contains definitional structure
            has_definition_structure = ':' in claim.claim_text or 'is' in claim.claim_text.lower()
            
            if not has_definition_structure:
                logger.debug(
                    f"Definition claim lacks proper structure: "
                    f"{claim.claim_text[:50]}..."
                )
                return False
        
        # Equation claims must reference equations
        elif claim.claim_type == ClaimType.EQUATION:
            # Check for equation-related evidence
            has_equation_evidence = any(
                evidence.source_type == "equation"
                for evidence in claim.evidence_objects
            )
            
            if not has_equation_evidence and self.strict_mode:
                logger.debug(
                    f"Equation claim lacks equation evidence: "
                    f"{claim.claim_text[:50]}..."
                )
                return False
        
        # Example claims should have substantial content
        elif claim.claim_type == ClaimType.EXAMPLE:
            if len(claim.claim_text) < 50:
                logger.debug(
                    f"Example claim too short: {claim.claim_text[:50]}..."
                )
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validator statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "validation_count": self.validation_count,
            "promoted_count": self.promoted_count,
            "rejected_count": self.rejected_count,
            "low_confidence_count": self.validation_count - self.promoted_count - self.rejected_count,
            "promotion_rate": self.promoted_count / self.validation_count if self.validation_count > 0 else 0.0,
            "rejection_rate": self.rejected_count / self.validation_count if self.validation_count > 0 else 0.0,
            "config": {
                "verified_threshold": self.verified_threshold,
                "rejected_threshold": self.rejected_threshold,
                "min_evidence_count": self.min_evidence_count,
                "strict_mode": self.strict_mode
            }
        }
