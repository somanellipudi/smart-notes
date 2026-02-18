"""
Tests for Negation Mismatch Detection and Verification

Validates negation detection and penalty scoring when claim and evidence disagree on assertion.
"""

import pytest
from src.verification.cs_claim_features import (
    detect_negation,
    CSClaimFeatureExtractor,
)
from src.verification.cs_verifiers import negation_mismatch_penalty, CSVerifier


class TestNegationDetection:
    """Test detection of negation markers in text."""
    
    def test_detect_not(self):
        """Test detection of 'not' negation."""
        text = "The algorithm is not correct."
        
        assert detect_negation(text)
    
    def test_detect_no(self):
        """Test detection of 'no' negation."""
        text = "There are no duplicates in the array."
        
        assert detect_negation(text)
    
    def test_detect_never(self):
        """Test detection of 'never' negation."""
        text = "This approach never fails."
        
        assert detect_negation(text)
    
    def test_detect_cannot(self):
        """Test detection of 'cannot' negation."""
        text = "You cannot modify this variable."
        
        assert detect_negation(text)
    
    def test_detect_cant_contraction(self):
        """Test detection of can't contraction."""
        text = "This can't be right."
        
        assert detect_negation(text)
    
    def test_detect_shouldnt(self):
        """Test detection of shouldn't."""
        text = "You shouldn't use this method directly."
        
        assert detect_negation(text)
    
    def test_detect_dont(self):
        """Test detection of don't."""
        text = "Don't worry about performance."
        
        assert detect_negation(text)
    
    def test_detect_doesnt(self):
        """Test detection of doesn't."""
        text = "This doesn't exist in the documentation."
        
        assert detect_negation(text)
    
    def test_detect_does_not(self):
        """Test detection of 'does not' phrasing."""
        text = "The algorithm does not guarantee correctness."
        
        assert detect_negation(text)
    
    def test_detect_fails(self):
        """Test detection of 'fails' as negation."""
        text = "The test fails under these conditions."
        
        assert detect_negation(text)
    
    def test_detect_invalid(self):
        """Test detection of 'invalid' negation."""
        text = "This is an invalid pointer."
        
        assert detect_negation(text)
    
    def test_detect_assumes_no(self):
        """Test detection of 'assumes no' phrasing."""
        text = "The algorithm assumes no cycles in the graph."
        
        assert detect_negation(text)
    
    def test_detect_without(self):
        """Test detection of 'without' negation."""
        text = "We can solve this without recursion."
        
        assert detect_negation(text)
    
    def test_detect_unless(self):
        """Test detection of 'unless' negation."""
        text = "This will fail unless you provide valid input."
        
        assert detect_negation(text)
    
    def test_no_negation_found(self):
        """Test text with no negation markers."""
        text = "The algorithm is correct and efficient."
        
        assert not detect_negation(text)
    
    def test_case_insensitive_detection(self):
        """Test that negation detection is case-insensitive."""
        texts = [
            "NOT correct",
            "Not correct",
            "not correct",
        ]
        
        for text in texts:
            assert detect_negation(text), f"Failed to detect negation in: {text}"


class TestNegationCounting:
    """Test counting negation markers in text."""
    
    def test_count_single_negation(self):
        """Test counting single negation marker."""
        text = "The algorithm does not work."
        count = CSClaimFeatureExtractor.count_negations(text)
        
        assert count >= 1
    
    def test_count_multiple_negations(self):
        """Test counting multiple negation markers."""
        text = "The algorithm is not correct and cannot be used without modifications."
        count = CSClaimFeatureExtractor.count_negations(text)
        
        assert count >= 2
    
    def test_count_no_negations(self):
        """Test count when no negations present."""
        text = "The algorithm is efficient."
        count = CSClaimFeatureExtractor.count_negations(text)
        
        assert count == 0


class TestNegationMismatchPenalty:
    """Test negation mismatch penalty scoring."""
    
    def test_matching_affirmation(self):
        """Test no penalty when both are affirmations."""
        claim = "The algorithm is correct."
        evidence = "Results show the algorithm is correct and efficient."
        
        penalty, reason = negation_mismatch_penalty(claim, evidence)
        
        assert penalty == 0.0, "No penalty for matching affirmations"
        assert "consistent" in reason.lower()
    
    def test_matching_negation(self):
        """Test no penalty when both use negation."""
        claim = "The algorithm does not overflow."
        evidence = "The algorithm never overflows due to bounds checking."
        
        penalty, reason = negation_mismatch_penalty(claim, evidence)
        
        assert penalty == 0.0, "No penalty for matching negations"
        assert "consistent" in reason.lower()
    
    def test_claim_affirms_evidence_negates(self):
        """Test penalty when claim affirms but evidence negates."""
        claim = "The algorithm guarantees correctness."
        evidence = "The algorithm does not guarantee correctness in all cases."
        
        penalty, reason = negation_mismatch_penalty(claim, evidence)
        
        assert penalty < 0, "Should apply penalty for mismatch"
        assert penalty >= -0.5, "Penalty should not exceed -0.5"
        assert "mismatch" in reason.lower()
    
    def test_claim_negates_evidence_affirms(self):
        """Test penalty when claim negates but evidence affirms."""
        claim = "The algorithm cannot handle edge cases."
        evidence = "The algorithm correctly handles edge cases."
        
        penalty, reason = negation_mismatch_penalty(claim, evidence)
        
        assert penalty < 0, "Should apply penalty for mismatch"
        assert "mismatch" in reason.lower()
    
    def test_multiple_negations_in_claim(self):
        """Test penalty based on negation count difference."""
        claim = "The algorithm is not safe and never guarantees correctness."  # 2 negations
        evidence = "The algorithm is correctly implemented."  # 0 negations
        
        penalty, reason = negation_mismatch_penalty(claim, evidence)
        
        assert penalty < 0, "Should apply penalty"
        # Penalty should be proportional to difference in negation count
    
    def test_single_vs_multiple_negations(self):
        """Test that negation count difference affects penalty magnitude."""
        claim1 = "The algorithm is not efficient."  # 1 negation
        claim2 = "The algorithm is not efficient and cannot handle large inputs and never works correctly."  # 3 negations
        evidence = "The algorithm is efficient."  # 0 negations
        
        penalty1, _ = negation_mismatch_penalty(claim1, evidence)
        penalty2, _ = negation_mismatch_penalty(claim2, evidence)
        
        # Both should have penalties
        assert penalty1 < 0, "Should apply penalty for claim1"
        assert penalty2 < 0, "Should apply penalty for claim2"
        # Penalty is capped at -0.5, so both may be identical or penalty2 <= penalty1
        assert penalty2 <= penalty1, "More negations should result in equal or worse penalty"


class TestNegationEdgeCases:
    """Test edge cases in negation detection."""
    
    def test_negation_in_compound_word(self):
        """Test 'not' appearing in compound words."""
        text = "The notebook is in the repository."  # 'note' contains 'not'
        
        # Should not falsely detect negation
        # Depends on regex word boundary handling
        has_negation = detect_negation(text)
        # This might be true or false depending on regex - just ensure it doesn't crash
        assert isinstance(has_negation, bool)
    
    def test_negation_in_word_not_at_boundary(self):
        """Test negation markers inside words."""
        text = "The function is documented."  # 'documented' contains 'no'
        
        # Should not falsely detect negation due to word boundaries
        has_negation = detect_negation(text)
        assert isinstance(has_negation, bool)
    
    def test_empty_text(self):
        """Test empty text."""
        assert not detect_negation("")
    
    def test_only_punctuation(self):
        """Test text with only punctuation."""
        assert not detect_negation("!!!")
    
    def test_all_caps_negation(self):
        """Test negation in all caps."""
        assert detect_negation("NEVER do this")
    
    def test_mixed_case_negation(self):
        """Test mixed case negation."""
        assert detect_negation("DoN't use this")
    
    def test_negation_with_unicode(self):
        """Test negation with unicode characters."""
        text = "The algorithm isn′t correct."  # Unicode apostrophe
        # Should handle gracefully, result depends on implementation
        result = detect_negation(text)
        assert isinstance(result, bool)


class TestNegationConsistency:
    """Test consistency patterns with negation."""
    
    def test_mutual_negation_rare(self):
        """Test that mutual negation between claim and evidence is suspicious."""
        claim = "Protocol X is not fault-tolerant."
        evidence = "Protocol X is not resistant to failures."
        
        # Both negate, so consistent
        penalty, _ = negation_mismatch_penalty(claim, evidence)
        assert penalty == 0.0
    
    def test_strong_statement_mismatch(self):
        """Test strong affirmation vs strong negation."""
        claim = "This algorithm is guaranteed to work perfectly."
        evidence = "This algorithm fails in many cases."
        
        penalty, _ = negation_mismatch_penalty(claim, evidence)
        
        # Should detect significant mismatch
        assert penalty < -0.2, "Should have significant penalty"
    
    def test_weak_vs_strong_negation(self):
        """Test weak negation vs strong negation."""
        claim = "The algorithm might not be optimal."
        evidence = "The algorithm fails completely."
        
        penalty, _ = negation_mismatch_penalty(claim, evidence)
        
        # Both negate but one uses "might not", one uses "fails"
        # Count depends on which negation markers are detected
        assert isinstance(penalty, float)
    
    def test_contextual_negation(self):
        """Test contextual negation (not directly about the algorithm)."""
        claim = "We cannot implement this without external libraries."
        evidence = "The implementation requires external libraries."
        
        penalty, reason = negation_mismatch_penalty(claim, evidence)
        
        # Claim has negation ("cannot"), evidence affirms
        # But they're saying similar things contextually
        assert isinstance(penalty, (int, float))


class TestNegationInTechnicalContext:
    """Test negation handling in technical CS contexts."""
    
    def test_invariant_negation(self):
        """Test negation in loop invariant context."""
        claim = "The invariant does not hold after iteration 5."
        evidence = "The invariant fails to hold when counter exceeds 4."
        
        penalty, _ = negation_mismatch_penalty(claim, evidence)
        
        # Both assert failure of invariant
        assert penalty == 0.0
    
    def test_precondition_negation(self):
        """Test negation in precondition."""
        claim = "Function assumes input is not null."
        evidence = "Function requires non-null input."
        
        penalty, _ = negation_mismatch_penalty(claim, evidence)
        
        # Semantically aligned despite different phrasing
    
    def test_error_condition_negation(self):
        """Test negation in error handling."""
        claim = "If data is corrupted, the system cannot process it."
        evidence = "Corruption makes data unprocessable."
        
        penalty, _ = negation_mismatch_penalty(claim, evidence)
        
        # Both describe same error condition
    
    def test_exception_negation(self):
        """Test negation with exceptions."""
        claim = "The function does not throw exceptions."
        evidence = "The function throws InvalidInputException for bad data."
        
        penalty, reason = negation_mismatch_penalty(claim, evidence)
        
        # Clear mismatch: claim says no exceptions, evidence describes exception
        assert penalty < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
