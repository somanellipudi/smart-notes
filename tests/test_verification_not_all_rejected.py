"""
Test verification not resulting in 100% rejection with matching sources.
"""

import pytest
from src.claims.schema import (
    LearningClaim,
    EvidenceItem,
    VerificationStatus,
    RejectionReason
)
from src.policies.evidence_policy import evaluate_evidence_sufficiency


def test_verification_not_all_rejected_when_sources_match():
    """
    Test that at least one claim verifies when good evidence is present.
    
    This is a fundamental sanity check: if evidence matches claim text with
    high similarity/entailment, the claim should NOT be rejected.
    
    Input:
        Claim: "Velocity is the rate of change of position."
        Evidence: Same text from transcript + notes (2 independent sources)
        Entailment: High probability (e.g., 0.85)
    
    Expected:
        Status should be VERIFIED (not REJECTED or LOW_CONFIDENCE)
    """
    # Setup
    claim = LearningClaim(
        claim_id="test_001",
        claim_type="definition",
        claim_text="Velocity is the rate of change of position."
    )
    
    evidence_items = [
        EvidenceItem(
            source_id="transcript_001",
            source_type="transcript",
            snippet="Velocity is the rate of change of position.",
            similarity=0.95
        ),
        EvidenceItem(
            source_id="notes_001",
            source_type="notes",
            snippet="Rate of change of position = velocity",
            similarity=0.88
        )
    ]
    
    nli_results = [
        {"label": "entailment", "score": 0.85},
        {"label": "entailment", "score": 0.82},
        {"label": "neutral", "score": 0.10},
        {"label": "contradiction", "score": 0.05}
    ]
    
    # Execute with standard thresholds
    decision = evaluate_evidence_sufficiency(
        claim=claim,
        evidence_items=evidence_items,
        nli_results=nli_results,
        min_entailment_prob=0.60,
        min_supporting_sources=2,
        max_contradiction_prob=0.30
    )
    
    # Assert
    assert decision.status_override == VerificationStatus.VERIFIED, (
        f"Expected VERIFIED but got {decision.status_override}. "
        f"Reason: {decision.reason}"
    )
    assert decision.independent_sources >= 2
    assert decision.confidence_score >= 0.60


def test_single_source_results_in_low_confidence():
    """Test that single source (even with good evidence) returns LOW_CONFIDENCE."""
    claim = LearningClaim(
        claim_id="test_002",
        claim_type="definition",
        claim_text="Force equals mass times acceleration."
    )
    
    # Only 1 independent source
    evidence_items = [
        EvidenceItem(
            source_id="transcript_001",
            source_type="transcript",
            snippet="Force equals mass times acceleration.",
            similarity=0.92
        )
    ]
    
    nli_results = [
        {"label": "entailment", "score": 0.88},
        {"label": "neutral", "score": 0.08},
        {"label": "contradiction", "score": 0.04}
    ]
    
    decision = evaluate_evidence_sufficiency(
        claim=claim,
        evidence_items=evidence_items,
        nli_results=nli_results,
        min_entailment_prob=0.60,
        min_supporting_sources=2,
        max_contradiction_prob=0.30
    )
    
    # Should be LOW_CONFIDENCE due to insufficient sources
    assert decision.status_override == VerificationStatus.LOW_CONFIDENCE
    assert "insufficient sources" in decision.reason.lower() or "insufficient" in decision.reason.lower()
    assert decision.independent_sources == 1


def test_low_entailment_rejection():
    """Test that low entailment probability results in REJECTED."""
    claim = LearningClaim(
        claim_id="test_003",
        claim_type="definition",
        claim_text="Quantum tunneling occurs in semiconductors."
    )
    
    evidence_items = [
        EvidenceItem(
            source_id="transcript_001",
            source_type="transcript",
            snippet="Semiconductors have band gaps.",
            similarity=0.45
        ),
        EvidenceItem(
            source_id="notes_001",
            source_type="notes",
            snippet="Tunneling is a phenomenon.",
            similarity=0.40
        )
    ]
    
    nli_results = [
        {"label": "entailment", "score": 0.35},  # Below threshold
        {"label": "neutral", "score": 0.50},
        {"label": "contradiction", "score": 0.15}
    ]
    
    decision = evaluate_evidence_sufficiency(
        claim=claim,
        evidence_items=evidence_items,
        nli_results=nli_results,
        min_entailment_prob=0.60,
        min_supporting_sources=2,
        max_contradiction_prob=0.30
    )
    
    # Should be REJECTED due to insufficient entailment
    assert decision.status_override == VerificationStatus.REJECTED
    assert "entailment" in decision.reason.lower()


def test_high_contradiction_results_in_low_confidence():
    """Test that high contradiction probability results in LOW_CONFIDENCE."""
    claim = LearningClaim(
        claim_id="test_004",
        claim_type="equation",
        claim_text="Quicksort has O(n²) worst-case complexity."
    )
    
    evidence_items = [
        EvidenceItem(
            source_id="textbook_001",
            source_type="external_context",
            snippet="Quicksort worst case is O(n²).",
            similarity=0.88
        ),
        EvidenceItem(
            source_id="notes_002",
            source_type="notes",
            snippet="Quicksort is O(n log n) in best case.",
            similarity=0.82
        )
    ]
    
    nli_results = [
        {"label": "entailment", "score": 0.75},
        {"label": "contradiction", "score": 0.55},  # High contradiction
        {"label": "neutral", "score": 0.15}
    ]
    
    decision = evaluate_evidence_sufficiency(
        claim=claim,
        evidence_items=evidence_items,
        nli_results=nli_results,
        min_entailment_prob=0.60,
        min_supporting_sources=2,
        max_contradiction_prob=0.30
    )
    
    # Should be LOW_CONFIDENCE due to high contradiction
    assert decision.status_override == VerificationStatus.LOW_CONFIDENCE
    assert "contradiction" in decision.reason.lower() or "conflict" in decision.reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
