"""
Unit tests for evidence sufficiency policy.

Tests cover:
- Independent source counting
- Evidence sufficiency evaluation
- Policy application to claims
"""

import pytest
from datetime import datetime

from src.policies.evidence_policy import (
    count_independent_sources,
    evaluate_evidence_sufficiency,
    apply_sufficiency_policy,
    SufficiencyDecision
)
from src.claims.schema import (
    LearningClaim,
    EvidenceItem,
    ClaimType,
    VerificationStatus,
    RejectionReason
)


class TestIndependentSourceCounting:
    """Test independent source counting logic."""
    
    def test_no_evidence(self):
        """No evidence should return 0 sources."""
        assert count_independent_sources([]) == 0
    
    def test_single_source(self):
        """Single evidence item."""
        evidence = [
            EvidenceItem(
                source_id="lecture1",
                source_type="transcript",
                snippet="Force equals mass times acceleration"
            )
        ]
        assert count_independent_sources(evidence) == 1
    
    def test_multiple_sources_different_ids(self):
        """Multiple sources with different IDs."""
        evidence = [
            EvidenceItem(source_id="lecture1", source_type="transcript", snippet="Force equals mass times acceleration"),
            EvidenceItem(source_id="lecture2", source_type="transcript", snippet="F equals m times a"),
        ]
        assert count_independent_sources(evidence) == 2
    
    def test_multiple_sources_same_id_different_types(self):
        """Same source ID but different types count as independent."""
        evidence = [
            EvidenceItem(source_id="lecture1", source_type="transcript", snippet="Force equals mass times acceleration"),
            EvidenceItem(source_id="lecture1", source_type="notes", snippet="F equals m times a"),
        ]
        assert count_independent_sources(evidence) == 2
    
    def test_duplicate_sources(self):
        """Duplicate (source_id, source_type) should count as one."""
        evidence = [
            EvidenceItem(source_id="lec", source_type="transcript", snippet="Force equals mass times acceleration"),
            EvidenceItem(source_id="lec", source_type="transcript", snippet="F equals m times a again"),
            EvidenceItem(source_id="lec", source_type="transcript", snippet="F equals m times a third time"),
        ]
        assert count_independent_sources(evidence) == 1


class TestEvidenceSufficiencyEvaluation:
    """Test evidence sufficiency decision rules."""
    
    def test_no_evidence_rejected(self):
        """No evidence should return REJECTED."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim",
            confidence=0.0
        )
        
        decision = evaluate_evidence_sufficiency(claim, [])
        
        assert decision.status_override == VerificationStatus.REJECTED
        assert decision.support_count == 0
        assert decision.independent_sources == 0
        assert "No evidence" in decision.reason
    
    def test_sufficient_evidence_verified(self):
        """Sufficient evidence should return VERIFIED."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Force equals mass times acceleration",
            confidence=0.0
        )
        
        evidence = [
            EvidenceItem(
                source_id="lec1",
                source_type="transcript",
                snippet="Force equals mass times acceleration",
                similarity=0.95
            ),
            EvidenceItem(
                source_id="notes",
                source_type="notes",
                snippet="Newton's second law: F = m * a",
                similarity=0.85
            )
        ]
        
        decision = evaluate_evidence_sufficiency(
            claim,
            evidence,
            min_entailment_prob=0.60,
            min_supporting_sources=2,
            max_contradiction_prob=0.30
        )
        
        assert decision.status_override == VerificationStatus.VERIFIED
        assert decision.support_count == 2
        assert decision.independent_sources == 2
        assert decision.confidence_score >= 0.60
    
    def test_insufficient_sources_low_confidence(self):
        """Insufficient independent sources should return LOW_CONFIDENCE."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim",
            confidence=0.0
        )
        
        evidence = [
            EvidenceItem(
                source_id="lec1",
                source_type="transcript",
                snippet="This is evidence text for the claim",
                similarity=0.80
            )
        ]
        
        decision = evaluate_evidence_sufficiency(
            claim,
            evidence,
            min_entailment_prob=0.60,
            min_supporting_sources=2,  # Requires 2, but only have 1
            max_contradiction_prob=0.30
        )
        
        assert decision.status_override == VerificationStatus.LOW_CONFIDENCE
        assert decision.independent_sources == 1
        assert "Insufficient independent sources" in decision.reason
    
    def test_low_entailment_rejected(self):
        """Low entailment probability should return REJECTED."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim",
            confidence=0.0
        )
        
        evidence = [
            EvidenceItem(
                source_id="lec1",
                source_type="transcript",
                snippet="Barely related text",
                similarity=0.30  # Low similarity
            ),
            EvidenceItem(
                source_id="lec2",
                source_type="notes",
                snippet="Also barely related",
                similarity=0.35
            )
        ]
        
        decision = evaluate_evidence_sufficiency(
            claim,
            evidence,
            min_entailment_prob=0.60,  # Requires 0.60, but only have 0.35
            min_supporting_sources=2,
            max_contradiction_prob=0.30
        )
        
        assert decision.status_override == VerificationStatus.REJECTED
        assert "Insufficient entailment" in decision.reason
    
    def test_contradiction_low_confidence(self):
        """High contradiction probability should return LOW_CONFIDENCE."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim",
            confidence=0.0
        )
        
        evidence = [
            EvidenceItem(
                source_id="lec1",
                source_type="transcript",
                snippet="This is supporting evidence for the claim",
                similarity=0.80
            ),
            EvidenceItem(
                source_id="lec2",
                source_type="notes",
                snippet="Additional supporting evidence text",
                similarity=0.75
            )
        ]
        
        # Mock NLI results with contradiction
        nli_results = [
            {"label": "entailment", "score": 0.70},
            {"label": "contradiction", "score": 0.40}  # High contradiction
        ]
        
        decision = evaluate_evidence_sufficiency(
            claim,
            evidence,
            nli_results=nli_results,
            min_entailment_prob=0.60,
            min_supporting_sources=2,
            max_contradiction_prob=0.30
        )
        
        assert decision.status_override == VerificationStatus.LOW_CONFIDENCE
        assert "contradiction" in decision.reason.lower()

    def test_adaptive_threshold_rejects_below_percentile(self):
        """Adaptive entailment threshold should reject low max entailment."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim",
            confidence=0.0
        )

        evidence = [
            EvidenceItem(
                source_id="lec1",
                source_type="transcript",
                snippet="Supporting text",
                similarity=0.55
            )
        ]

        nli_results = [
            {"label": "entailment", "score": 0.55},
            {"label": "contradiction", "score": 0.05}
        ]

        distribution = [0.4, 0.5, 0.6, 0.7, 0.9]

        decision = evaluate_evidence_sufficiency(
            claim,
            evidence,
            nli_results=nli_results,
            entailment_score_distribution=distribution,
            min_supporting_sources=1
        )

        assert decision.entailment_threshold_used >= 0.50
        assert decision.status_override == VerificationStatus.REJECTED

    def test_adaptive_threshold_allows_above_percentile(self):
        """Adaptive entailment threshold should allow when max entailment is high enough."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim",
            confidence=0.0
        )

        evidence = [
            EvidenceItem(
                source_id="lec1",
                source_type="transcript",
                snippet="Supporting text",
                similarity=0.62
            )
        ]

        nli_results = [
            {"label": "entailment", "score": 0.62},
            {"label": "contradiction", "score": 0.05}
        ]

        distribution = [0.4, 0.5, 0.6, 0.7, 0.9]

        decision = evaluate_evidence_sufficiency(
            claim,
            evidence,
            nli_results=nli_results,
            entailment_score_distribution=distribution,
            min_supporting_sources=1
        )

        assert decision.entailment_threshold_used >= 0.50
        assert decision.status_override == VerificationStatus.VERIFIED

    def test_single_source_system_allows_one_source(self):
        """Allow single source when only one independent source exists system-wide."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim",
            confidence=0.0
        )

        evidence = [
            EvidenceItem(
                source_id="lec1",
                source_type="transcript",
                snippet="Supporting text",
                similarity=0.80
            )
        ]

        decision = evaluate_evidence_sufficiency(
            claim,
            evidence,
            min_entailment_prob=0.60,
            min_supporting_sources=2,
            total_independent_sources=1
        )

        assert decision.required_sources == 1
        assert decision.status_override == VerificationStatus.VERIFIED


class TestPolicyApplication:
    """Test policy application to claims."""
    
    def test_apply_policy_updates_claim(self):
        """Policy application should update claim status and confidence."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim",
            confidence=0.0,
            status=VerificationStatus.REJECTED
        )
        
        evidence = [
            EvidenceItem(
                source_id="lec1",
                source_type="transcript",
                snippet="This is supporting evidence for the claim",
                similarity=0.85
            ),
            EvidenceItem(
                source_id="notes",
                source_type="notes",
                snippet="Additional supporting evidence text",
                similarity=0.80
            )
        ]
        
        apply_sufficiency_policy(claim, evidence)
        
        # Claim should be updated
        assert claim.status == VerificationStatus.VERIFIED
        assert claim.confidence > 0.60
        assert claim.rejection_reason is None
        assert "sufficiency_decision" in claim.metadata
    
    def test_apply_policy_rejection(self):
        """Policy should set rejection reason for rejected claims."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim",
            confidence=0.0,
            status=VerificationStatus.REJECTED
        )
        
        # No evidence
        apply_sufficiency_policy(claim, [])
        
        assert claim.status == VerificationStatus.REJECTED
        assert claim.rejection_reason == RejectionReason.NO_EVIDENCE
        assert claim.confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
