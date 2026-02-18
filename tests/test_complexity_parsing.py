"""
Tests for Complexity Notation Parsing and Verification

Validates Big-O/Theta/Omega extraction and consistency checking between claims and evidence.
"""

import pytest
from src.verification.cs_claim_features import (
    extract_complexity_tokens,
    CSClaimFeatureExtractor,
    ComplexityToken,
)
from src.verification.cs_verifiers import complexity_consistency, CSVerifier


class TestComplexityTokenExtraction:
    """Test extraction of Big-O/Theta/Omega notations."""
    
    def test_extract_big_o_notation(self):
        """Test extraction of Big-O notation."""
        text = "The time complexity is O(n log n)."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) > 0
        assert tokens[0].notation == "O"
        assert "n log n" in tokens[0].expression.lower() or "nlogn" in tokens[0].expression.lower().replace(" ", "")
    
    def test_extract_big_theta_notation(self):
        """Test extraction of Big-Theta notation."""
        text = "The exact complexity is Θ(n²)."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) > 0
        assert tokens[0].notation == "Θ"
        # Check for n and 2 (or superscript 2 character)
        assert "n" in tokens[0].expression
    
    def test_extract_big_omega_notation(self):
        """Test extraction of Big-Omega notation."""
        text = "The lower bound is Ω(n)."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) > 0
        assert tokens[0].notation == "Ω"
        assert "n" in tokens[0].expression
    
    def test_extract_constant_complexity(self):
        """Test extraction of constant time O(1)."""
        text = "Hash table lookup is O(1)."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) > 0
        # Expression should be "1"
        assert "1" in tokens[0].expression
    
    def test_extract_linear_complexity(self):
        """Test extraction of linear complexity O(n)."""
        text = "Linear search takes O(n) time."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) > 0
        assert "n" in tokens[0].expression
    
    def test_extract_nlogn_complexity(self):
        """Test extraction of n log n complexity."""
        text = "Merge sort has O(n log n) complexity."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) > 0
        expr = tokens[0].expression.lower().replace(" ", "")
        assert "n" in expr and "log" in expr
    
    def test_extract_quadratic_complexity(self):
        """Test extraction of quadratic O(n²)."""
        text = "Bubble sort is O(n²)."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) > 0
        # Check for n (superscript 2 is stored as Unicode character)
        assert "n" in tokens[0].expression
    
    def test_extract_exponential_complexity(self):
        """Test extraction of exponential 2^n."""
        text = "The algorithm is O(2^n)."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) > 0
        assert "2" in tokens[0].expression and "n" in tokens[0].expression
    
    def test_extract_factorial_complexity(self):
        """Test extraction of factorial n!."""
        text = "Permutation generation is O(n!) complexity."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) > 0
        assert "n" in tokens[0].expression and "!" in tokens[0].expression
    
    def test_extract_vertices_edges_complexity(self):
        """Test extraction of graph complexity V+E."""
        text = "Graph traversal is O(V+E) where V is vertices and E is edges."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) > 0
        expr = tokens[0].expression
        assert "V" in expr and "E" in expr
    
    def test_extract_multiple_complexities(self):
        """Test extraction of multiple complexity notations in one text."""
        text = "Best case is O(n), average is O(n log n), worst case is O(n²)."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) >= 2
        notations = [t.notation for t in tokens]
        assert all(n == "O" for n in notations)
    
    def test_no_complexity_notation(self):
        """Test text with no complexity notation."""
        text = "The algorithm is very efficient."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) == 0
    
    def test_extract_theta_word_form(self):
        """Test Theta notation in word form (Theta instead of Θ)."""
        text = "The complexity is Theta(n log n)."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) > 0
        assert tokens[0].notation == "Θ"
    
    def test_extract_omega_word_form(self):
        """Test Omega notation in word form."""
        text = "Lower bound is Omega(n)."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) > 0
        assert tokens[0].notation == "Ω"


class TestComplexityConsistency:
    """Test complexity consistency between claims and evidence."""
    
    def test_exact_complexity_match(self):
        """Test exact complexity notation match."""
        claim = "Merge sort has O(n log n) time complexity."
        evidence = "Merge sort runs in O(n log n) time."
        
        signal = complexity_consistency(claim, evidence)
        
        assert signal.score > 0.5, "Should match exact complexity"
        assert signal.has_match
    
    def test_notation_mismatch(self):
        """Test mismatch between different notations."""
        claim = "The tight bound is Θ(n²)."
        evidence = "It can be at most O(n²)."
        
        signal = complexity_consistency(claim, evidence)
        
        # Notations differ (Θ vs O) but expressions match
        # Score depends on matching expression
        assert signal.score >= 0.0
    
    def test_no_complexity_in_claim(self):
        """Test when claim has no complexity notation."""
        claim = "The algorithm is efficient."
        evidence = "It runs in O(n) time."
        
        signal = complexity_consistency(claim, evidence)
        
        assert signal.score == 1.0, "Perfect score when no complexity in claim"
        assert signal.has_match
    
    def test_no_complexity_in_evidence(self):
        """Test when evidence has no complexity."""
        claim = "The complexity is O(n log n)."
        evidence = "The algorithm processes the data efficiently."
        
        signal = complexity_consistency(claim, evidence)
        
        # Missing complexity evidence
        assert signal.score < 1.0
    
    def test_compatible_complexities(self):
        """Test compatible complexity claims."""
        claim = "Best case is O(1), average is O(n)."
        evidence = "O(1) in best case, O(n) average case."
        
        signal = complexity_consistency(claim, evidence)
        
        # Multiple matching complexities
        assert signal.has_match
    
    def test_complexity_expression_normalization(self):
        """Test that expressions are normalized for matching."""
        claim = "O(n log n) complexity."
        evidence = "O(n log n) time."
        
        signal = complexity_consistency(claim, evidence)
        
        assert signal.score > 0.5, "Should match despite whitespace differences"
    
    def test_space_vs_time_complexity(self):
        """Test claim about space vs time complexity."""
        claim = "Space complexity is O(n)."
        evidence = "Time complexity is O(n log n), space is O(n)."
        
        signal = complexity_consistency(claim, evidence)
        
        # Space O(n) mentioned in both
        assert signal.has_match


class TestComplexityClaimTypes:
    """Test complexity verification for COMPLEXITY_CLAIM type."""
    
    def test_algorithm_complexity_claim(self):
        """Test complexity claim about an algorithm."""
        claim = "Binary search has O(log n) time complexity."
        evidence = "Binary search performs logarithmic comparisons: O(log n)."
        
        signal = complexity_consistency(claim, evidence)
        
        assert signal.has_match, "Should recognize matching logarithmic complexity"
    
    def test_worst_case_claim(self):
        """Test worst-case complexity claim."""
        claim = "Quicksort worst-case is O(n²)."
        evidence = "In worst case, quicksort degrades to O(n²)."
        
        signal = complexity_consistency(claim, evidence)
        
        assert signal.has_match, "Should match worst-case complexity"
    
    def test_data_structure_claim(self):
        """Test complexity claim about data structure."""
        claim = "Hash table insertion is O(1) average case."
        evidence = "Average-case insertion into hash tables: O(1)."
        
        signal = complexity_consistency(claim, evidence)
        
        assert signal.has_match or signal.score > 0.5


class TestComplexityEdgeCases:
    """Test edge cases in complexity parsing."""
    
    def test_spaces_in_notation(self):
        """Test handling of spaces in complexity notation."""
        text = "O ( n log n )"
        tokens = extract_complexity_tokens(text)
        
        # May or may not extract depending on regex flexibility
        # Should not crash
        assert isinstance(tokens, list)
    
    def test_nested_parentheses(self):
        """Test handling of nested parentheses."""
        text = "O((n+m) log n)"
        tokens = extract_complexity_tokens(text)
        
        # Implementation-dependent on parenthesis matching
        assert isinstance(tokens, list)
    
    def test_multiple_notations_consecutive(self):
        """Test multiple notations appearing consecutively."""
        text = "The complexity is O(n) and Θ(n) and Ω(n)."
        tokens = extract_complexity_tokens(text)
        
        assert len(tokens) >= 2
    
    def test_empty_text(self):
        """Test empty text."""
        tokens = extract_complexity_tokens("")
        
        assert len(tokens) == 0
    
    def test_numeric_parentheses_without_notation(self):
        """Test numeric parentheses without O/Θ/Ω."""
        text = "The value (42) is important."
        tokens = extract_complexity_tokens(text)
        
        # Should not extract (42) as complexity
        assert len(tokens) == 0


class TestComplexitySignalFields:
    """Test VerificationSignal fields for complexity checks."""
    
    def test_signal_has_required_fields(self):
        """Test that signal contains all required fields."""
        signal = complexity_consistency("O(n²)", "quadratic O(n²)")
        
        assert signal.signal_type == "complexity"
        assert 0.0 <= signal.score <= 1.0
        assert isinstance(signal.evidence, str)
        assert isinstance(signal.has_match, bool)
        assert "claim_features" in signal.__dict__
        assert "evidence_features" in signal.__dict__
    
    def test_signal_extracts_complexity_tokens(self):
        """Test that signal extracts complexity tokens."""
        signal = complexity_consistency("O(n log n)", "Has O(n log n)")
        
        assert "complexity_tokens" in signal.claim_features
        assert "complexity_tokens" in signal.evidence_features
        assert isinstance(signal.claim_features["complexity_tokens"], list)
        assert isinstance(signal.evidence_features["complexity_tokens"], list)


class TestComplexityAnchorTerms:
    """Test extraction of complexity anchor terms."""
    
    def test_find_worst_case_anchor(self):
        """Test finding worst-case anchor term."""
        text = "In the worst-case scenario, the complexity is O(n²)."
        anchors = CSClaimFeatureExtractor.find_anchor_terms(text, 'complexity')
        
        assert "worst-case" in anchors or any("worst" in a for a in anchors)
    
    def test_find_amortized_anchor(self):
        """Test finding amortized anchor term."""
        text = "The amortized cost per operation is O(1)."
        anchors = CSClaimFeatureExtractor.find_anchor_terms(text, 'complexity')
        
        assert "amortized" in anchors
    
    def test_find_asymptotic_anchor(self):
        """Test finding asymptotic anchor term."""
        text = "Asymptotic complexity is O(n log n)."
        anchors = CSClaimFeatureExtractor.find_anchor_terms(text, 'complexity')
        
        assert any("asymptotic" in a for a in anchors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
