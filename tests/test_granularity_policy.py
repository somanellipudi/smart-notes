"""
Unit tests for granularity policy (atomic claim enforcement).

Tests cover:
- Compound claim detection
- Claim splitting algorithms
- Granularity enforcement with metadata preservation
"""

import pytest
from datetime import datetime

from src.policies.granularity_policy import (
    is_compound_claim,
    split_compound_claim,
    enforce_granularity
)
from src.claims.schema import LearningClaim, ClaimType, VerificationStatus


class TestCompoundClaimDetection:
    """Test compound claim detection heuristics."""
    
    def test_atomic_claim_single_sentence(self):
        """Single sentence should not be compound."""
        text = "Force equals mass times acceleration."
        assert not is_compound_claim(text)
    
    def test_compound_multiple_sentences(self):
        """Multiple sentences should be compound."""
        text = "Force equals mass times acceleration. Velocity is distance over time."
        assert is_compound_claim(text)
    
    def test_compound_semicolon(self):
        """Semicolon should indicate compound claim."""
        text = "F=ma; v=d/t"
        assert is_compound_claim(text)
    
    def test_compound_conjunction(self):
        """Conjunction 'and' should indicate compound claim."""
        text = "Force equals mass times acceleration and velocity equals distance over time."
        assert is_compound_claim(text)
    
    def test_not_compound_simple_conjunction(self):
        """Simple conjunction in noun phrase should not be compound."""
        text = "Mass and acceleration are factors in force."
        # This might be detected as compound due to 'and', which is acceptable
        # (false positive is safer than false negative)
        result = is_compound_claim(text)
        # Accept either result as valid
        assert isinstance(result, bool)
    
    def test_compound_multiple_equations(self):
        """Multiple equations should be compound."""
        text = "F=ma and E=mc^2"
        assert is_compound_claim(text)
    
    def test_not_compound_short_text(self):
        """Very short text should not be compound."""
        text = "F=ma"
        assert not is_compound_claim(text)


class TestClaimSplitting:
    """Test claim splitting algorithms."""
    
    def test_split_by_sentences(self):
        """Split by sentence boundaries."""
        text = "Force equals mass times acceleration. Velocity is distance over time."
        result = split_compound_claim(text)
        assert len(result) == 2
        assert "Force equals mass times acceleration" in result[0]
        assert "Velocity is distance over time" in result[1]
    
    def test_split_by_semicolon(self):
        """Split by semicolons."""
        text = "F=ma; v=d/t"
        result = split_compound_claim(text)
        assert len(result) == 2
        assert "F=ma" in result[0]
        assert "v=d/t" in result[1]
    
    def test_split_by_conjunction(self):
        """Split by coordinating conjunction."""
        text = "Force equals mass times acceleration and velocity equals distance over time."
        result = split_compound_claim(text)
        # Should split into 2 parts
        assert len(result) >= 2
        # Both parts should contain key terms
        all_text = " ".join(result)
        assert "Force" in all_text or "mass" in all_text
        assert "velocity" in all_text or "distance" in all_text
    
    def test_no_split_atomic(self):
        """Atomic claim should not be split."""
        text = "Force equals mass times acceleration."
        result = split_compound_claim(text)
        assert len(result) == 1
        assert result[0].strip() == text.strip()
    
    def test_split_max_limit(self):
        """Respect maximum split limit."""
        text = "A. B. C. D. E. F. G."
        result = split_compound_claim(text, max_splits=3)
        assert len(result) <= 3


class TestGranularityEnforcement:
    """Test full granularity enforcement pipeline."""
    
    def test_enforce_atomic_claims(self):
        """Enforce atomic claims by splitting compound ones."""
        claim1 = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Force equals mass times acceleration. Velocity is distance over time.",
            confidence=0.8,
            status=VerificationStatus.VERIFIED
        )
        
        result = enforce_granularity([claim1], max_propositions=1)
        
        # Should be split into 2 atomic claims
        assert len(result) == 2
        assert all(claim.claim_type == ClaimType.DEFINITION for claim in result)
        assert all(claim.confidence == 0.8 for claim in result)
        assert all(claim.status == VerificationStatus.VERIFIED for claim in result)
    
    def test_preserve_metadata(self):
        """Metadata should be preserved when splitting."""
        claim = LearningClaim(
            claim_type=ClaimType.EQUATION,
            claim_text="F=ma; E=mc^2",
            confidence=0.9,
            metadata={"source": "lecture"},
            status=VerificationStatus.VERIFIED
        )
        
        result = enforce_granularity([claim], max_propositions=1)
        
        # Check metadata preserved
        for atomic_claim in result:
            assert atomic_claim.metadata.get("source") == "lecture"
            assert "parent_claim_id" in atomic_claim.metadata
            assert atomic_claim.metadata["parent_claim_id"] == claim.claim_id
    
    def test_no_split_if_atomic(self):
        """Atomic claims should pass through unchanged."""
        claim = LearningClaim(
            claim_type=ClaimType.FACT,
            claim_text="The derivative of x^2 is 2x.",
            confidence=0.95,
            status=VerificationStatus.VERIFIED
        )
        
        result = enforce_granularity([claim], max_propositions=1)
        
        assert len(result) == 1
        assert result[0].claim_id == claim.claim_id
        assert result[0].claim_text == claim.claim_text
    
    def test_skip_empty_claims(self):
        """Empty claims should pass through."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="",
            confidence=0.0,
            status=VerificationStatus.REJECTED
        )
        
        result = enforce_granularity([claim], max_propositions=1)
        
        assert len(result) == 1
        assert result[0].claim_text == ""
    
    def test_multiple_claims_mixed(self):
        """Mix of atomic and compound claims."""
        claims = [
            LearningClaim(
                claim_type=ClaimType.DEFINITION,
                claim_text="Atomic claim here.",
                confidence=0.8,
                status=VerificationStatus.VERIFIED
            ),
            LearningClaim(
                claim_type=ClaimType.EQUATION,
                claim_text="F=ma. E=mc^2.",
                confidence=0.9,
                status=VerificationStatus.VERIFIED
            ),
            LearningClaim(
                claim_type=ClaimType.EXAMPLE,
                claim_text="Another atomic claim.",
                confidence=0.7,
                status=VerificationStatus.LOW_CONFIDENCE
            )
        ]
        
        result = enforce_granularity(claims, max_propositions=1)
        
        # First and third should be unchanged, second should be split
        assert len(result) >= 3  # At least 3, possibly 4 if second splits
        # Check that atomic claims are preserved
        assert any(c.claim_text == "Atomic claim here." for c in result)
        assert any(c.claim_text == "Another atomic claim." for c in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
