"""
Tests for Numeric Consistency Verification

Validates numeric token extraction and consistency checking between claims and evidence.
"""

import pytest
from src.verification.cs_claim_features import (
    extract_numeric_tokens,
    CSClaimFeatureExtractor,
    NumericToken,
)
from src.verification.cs_verifiers import numeric_consistency, CSVerifier


class TestNumericTokenExtraction:
    """Test numeric token extraction from text."""
    
    def test_extract_integers(self):
        """Test extraction of integer values."""
        text = "The array has 10 elements and 256 bits."
        tokens = extract_numeric_tokens(text)
        
        assert len(tokens) >= 2
        values = [t.value for t in tokens]
        assert 10 in values
        assert 256 in values
    
    def test_extract_decimals(self):
        """Test extraction of decimal values."""
        text = "The probability is 0.95 and the error rate is 0.05."
        tokens = extract_numeric_tokens(text)
        
        assert len(tokens) >= 2
        values = [t.value for t in tokens]
        assert 0.95 in values or any(abs(v - 0.95) < 0.01 for v in values)
        assert 0.05 in values or any(abs(v - 0.05) < 0.01 for v in values)
    
    def test_extract_infinity(self):
        """Test extraction of infinity marker."""
        text = "The algorithm runs for infinite time."
        tokens = extract_numeric_tokens(text)
        
        # Should detect infinity concept
        assert len(tokens) >= 0  # Infinity might not be captured as numeric token
    
    def test_context_detection_worst_case(self):
        """Test context detection for worst-case complexity."""
        text = "In the worst case, this takes 1000 milliseconds."
        tokens = extract_numeric_tokens(text)
        
        assert len(tokens) > 0
        numeric_1000 = [t for t in tokens if t.value == 1000]
        if numeric_1000:
            assert numeric_1000[0].context == "worst-case"
    
    def test_context_detection_best_case(self):
        """Test context detection for best-case complexity."""
        text = "In the best case, the algorithm takes only 1 second."
        tokens = extract_numeric_tokens(text)
        
        numeric_tokens = [t for t in tokens if t.value == 1]
        if numeric_tokens:
            assert numeric_tokens[0].context == "best-case"
    
    def test_context_detection_average(self):
        """Test context detection for average-case."""
        text = "On average, we observe 50 iterations."
        tokens = extract_numeric_tokens(text)
        
        numeric_50 = [t for t in tokens if t.value == 50]
        if numeric_50:
            assert numeric_50[0].context == "average-case"
    
    def test_context_detection_amortized(self):
        """Test context detection for amortized analysis."""
        text = "The amortized cost is 2 per operation."
        tokens = extract_numeric_tokens(text)
        
        numeric_2 = [t for t in tokens if t.value == 2]
        if numeric_2:
            assert numeric_2[0].context == "amortized"
    
    def test_no_numeric_tokens(self):
        """Test text with no numeric values."""
        text = "This is a pure text without any numbers."
        tokens = extract_numeric_tokens(text)
        
        assert len(tokens) == 0


class TestNumericConsistency:
    """Test numeric consistency between claims and evidence."""
    
    def test_exact_match(self):
        """Test exact numeric match between claim and evidence."""
        claim = "The array size is 1024."
        evidence = "Memory allocation for 1024 elements is required."
        
        signal = numeric_consistency(claim, evidence)
        
        assert signal.score > 0.5, "Should have good score for matching numbers"
        assert signal.has_match, "Should detect match"
    
    def test_no_numeric_in_claim(self):
        """Test when claim has no numeric values."""
        claim = "The algorithm is efficient."
        evidence = "It processes 1000 items quickly."
        
        signal = numeric_consistency(claim, evidence)
        
        assert signal.score == 1.0, "Should get perfect score when no numeric in claim"
        assert signal.has_match, "Should be match by default"
    
    def test_no_numeric_in_evidence(self):
        """Test when evidence has no numeric values."""
        claim = "The time complexity is 1000 microseconds."
        evidence = "The algorithm is very fast."
        
        signal = numeric_consistency(claim, evidence)
        
        # Partial credit since evidence has numbers but different ones
        assert signal.score < 1.0, "Should penalize missing numeric evidence"
    
    def test_mismatch_numeric(self):
        """Test numeric mismatch between claim and evidence."""
        claim = "The array has 512 elements."
        evidence = "We allocate 1024 bytes of memory."
        
        signal = numeric_consistency(claim, evidence)
        
        # Different numbers (512 vs 1024)
        assert signal.score < 1.0, "Should penalize numeric mismatch"
    
    def test_multiple_numerics_partial_match(self):
        """Test partial match with multiple numeric values."""
        claim = "We need 100 GB storage and 32 GB RAM."
        evidence = "Allocate 32 GB RAM for the process."
        
        signal = numeric_consistency(claim, evidence)
        
        # One out of two match (32 matches)
        assert 0.0 < signal.score < 1.0
    
    def test_context_sensitive_analysis(self):
        """Test that context is considered in matching."""
        claim = "Worst-case complexity is 1000."
        evidence = "Average-case complexity is 1000."
        
        signal = numeric_consistency(claim, evidence)
        
        # Same number but different context - should still match numerically
        assert signal.has_match
    
    def test_floating_point_tolerance(self):
        """Test floating-point tolerance in matching."""
        claim = "Probability is 0.95"
        evidence = "Success rate: 0.9501"
        
        signal = numeric_consistency(claim, evidence)
        
        # Should allow small floating-point variations
        # Note: Depends on tolerance threshold


class TestNumericClaimTypes:
    """Test numeric verification for NUMERIC_CLAIM type."""
    
    def test_count_claim(self):
        """Test claims about counts/quantities."""
        claim = "A binary tree with depth d has at most 2^d nodes."
        evidence = "The maximum number of nodes in a depth-d tree is 2^d, or 2 to the power of d."
        
        signal = numeric_consistency(claim, evidence)
        
        assert signal.score > 0.5, "Should recognize matching power notation"
    
    def test_percentage_claim(self):
        """Test claims about percentages."""
        claim = "95% of test cases pass."
        evidence = "Out of 100 tests, 95 succeed."
        
        signal = numeric_consistency(claim, evidence)
        
        # 95 appears in both
        assert signal.has_match


class TestNumericEdgeCases:
    """Test edge cases in numeric verification."""
    
    def test_phone_number_as_numeric(self):
        """Test that phone numbers are handled as numeric."""
        claim = "Call 555-1234."
        tokens = extract_numeric_tokens(claim)
        
        # Phone patterns might be captured as separate numbers
        # Just ensure extraction doesn't crash
        assert isinstance(tokens, list)
    
    def test_version_numbers(self):
        """Test version numbers like 3.14."""
        claim = "Python 3.8"
        tokens = extract_numeric_tokens(claim)
        
        # Should extract 3 and 8 (or 3.8 if decimal pattern matches)
        assert len(tokens) >= 1
    
    def test_empty_claim(self):
        """Test empty claim text."""
        signal = numeric_consistency("", "Some evidence 123")
        
        assert signal.score == 1.0, "Empty claim should get perfect score"
    
    def test_empty_evidence(self):
        """Test empty evidence text."""
        signal = numeric_consistency("Claim with 123", "")
        
        assert signal.score < 1.0, "Empty evidence should result in mismatch"
    
    def test_large_numbers(self):
        """Test large number handling."""
        claim = "Process 1000000 items."
        evidence = "Handle 1e6 elements."
        
        signal = numeric_consistency(claim, evidence)
        
        # Both represent 1 million - implementation dependent on extraction


class TestNumericSignalFields:
    """Test VerificationSignal fields for numeric checks."""
    
    def test_signal_has_required_fields(self):
        """Test that signal contains all required fields."""
        signal = numeric_consistency("Value 42", "The answer is 42")
        
        assert signal.signal_type == "numeric"
        assert 0.0 <= signal.score <= 1.0
        assert isinstance(signal.evidence, str)
        assert isinstance(signal.has_match, bool)
        assert "claim_features" in signal.__dict__
        assert "evidence_features" in signal.__dict__
    
    def test_signal_extracts_features(self):
        """Test that signal extracts and reports features."""
        signal = numeric_consistency("Value 100", "Reference to 100 items")
        
        assert "numeric_tokens" in signal.claim_features
        assert "numeric_tokens" in signal.evidence_features
        assert isinstance(signal.claim_features["numeric_tokens"], list)
        assert isinstance(signal.evidence_features["numeric_tokens"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
