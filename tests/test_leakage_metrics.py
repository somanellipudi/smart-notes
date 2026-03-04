"""
Tests for leakage metrics computation.

Validates that LCO, LCS, and SUBSTRING metrics compute correct values
on known inputs and edge cases.
"""

import pytest
from scripts.leakage_scan import (
    tokenize,
    longest_consecutive_overlap,
    longest_common_subsequence_ratio,
    longest_common_substring_ratio,
)


# ============================================================================
# Tokenization Tests
# ============================================================================

@pytest.mark.paper
def test_tokenize_simple():
    """Test basic tokenization."""
    result = tokenize("Hello World")
    assert result == ["hello", "world"]


@pytest.mark.paper
def test_tokenize_lowercase():
    """Test that tokenization lowercases."""
    result = tokenize("HELLO")
    assert result == ["hello"]


@pytest.mark.paper
def test_tokenize_punctuation_removal():
    """Test that punctuation is removed."""
    result = tokenize("Hello, World!")
    assert result == ["hello", "world"]


@pytest.mark.paper
def test_tokenize_multiple_spaces():
    """Test that multiple spaces are handled."""
    result = tokenize("hello    world")
    assert result == ["hello", "world"]


# ============================================================================
# LCO (Longest Consecutive Overlap) Tests
# ============================================================================

@pytest.mark.paper
def test_lco_exact_match():
    """Test LCO when claim is substring of passage."""
    claim_tokens = ["the", "quick", "brown"]
    passage_tokens = ["the", "quick", "brown", "fox"]
    
    result = longest_consecutive_overlap(claim_tokens, passage_tokens)
    # All 3 tokens match consecutively
    assert result == pytest.approx(1.0, abs=0.001)


@pytest.mark.paper
def test_lco_partial_overlap():
    """Test LCO with partial overlap."""
    claim_tokens = ["the", "quick", "brown"]
    passage_tokens = ["the", "very", "quick", "brown", "fox"]
    
    result = longest_consecutive_overlap(claim_tokens, passage_tokens)
    # "the" and "quick" don't form consecutive substring in passage
    # but "quick brown" is consecutive (2 tokens out of 3)
    assert result == pytest.approx(2.0 / 3.0, abs=0.001)


@pytest.mark.paper
def test_lco_no_overlap():
    """Test LCO with no overlap."""
    claim_tokens = ["cat", "dog"]
    passage_tokens = ["bird", "fish"]
    
    result = longest_consecutive_overlap(claim_tokens, passage_tokens)
    assert result == pytest.approx(0.0, abs=0.001)


@pytest.mark.paper
def test_lco_single_token_match():
    """Test LCO with single token match."""
    claim_tokens = ["hello", "world"]
    passage_tokens = ["hello", "there"]
    
    result = longest_consecutive_overlap(claim_tokens, passage_tokens)
    # "hello" is the only match (1 out of 2)
    assert result == pytest.approx(0.5, abs=0.001)


@pytest.mark.paper
def test_lco_empty_claim_raises():
    """Test that empty claim raises ValueError."""
    with pytest.raises(ValueError):
        longest_consecutive_overlap([], ["some", "tokens"])


# ============================================================================
# LCS (Longest Common Subsequence) Tests
# ============================================================================

@pytest.mark.paper
def test_lcs_exact_match():
    """Test LCS when all tokens match."""
    claim_tokens = ["a", "b", "c"]
    passage_tokens = ["a", "b", "c"]
    
    result = longest_common_subsequence_ratio(claim_tokens, passage_tokens)
    assert result == pytest.approx(1.0, abs=0.001)


@pytest.mark.paper
def test_lcs_subsequence_only():
    """Test LCS when tokens match but not consecutively."""
    claim_tokens = ["a", "b", "c"]
    passage_tokens = ["a", "x", "b", "y", "c"]
    
    result = longest_common_subsequence_ratio(claim_tokens, passage_tokens)
    # All 3 tokens appear in passage (as subsequence)
    assert result == pytest.approx(1.0, abs=0.001)


@pytest.mark.paper
def test_lcs_partial():
    """Test LCS with partial match."""
    claim_tokens = ["a", "b", "c"]
    passage_tokens = ["a", "b", "z"]
    
    result = longest_common_subsequence_ratio(claim_tokens, passage_tokens)
    # "a" and "b" match (2 out of 3)
    assert result == pytest.approx(2.0 / 3.0, abs=0.001)


@pytest.mark.paper
def test_lcs_no_overlap():
    """Test LCS with no overlap."""
    claim_tokens = ["a", "b"]
    passage_tokens = ["x", "y"]
    
    result = longest_common_subsequence_ratio(claim_tokens, passage_tokens)
    assert result == pytest.approx(0.0, abs=0.001)


@pytest.mark.paper
def test_lcs_empty_claim_raises():
    """Test that empty claim raises ValueError."""
    with pytest.raises(ValueError):
        longest_common_subsequence_ratio([], ["some", "tokens"])


# ============================================================================
# SUBSTRING (Token Contiguous Substring) Tests
# ============================================================================

@pytest.mark.paper
def test_substring_exact_match():
    """Test SUBSTRING when claim is substring of passage."""
    claim_tokens = ["hello", "world"]
    passage_tokens = ["say", "hello", "world", "now"]
    
    result = longest_common_substring_ratio(claim_tokens, passage_tokens)
    # Both tokens match consecutively in passage
    assert result == pytest.approx(1.0, abs=0.001)


@pytest.mark.paper
def test_substring_partial_match():
    """Test SUBSTRING with partial contiguous match."""
    claim_tokens = ["a", "b", "c"]
    passage_tokens = ["a", "b", "x"]
    
    result = longest_common_substring_ratio(claim_tokens, passage_tokens)
    # "a" "b" match consecutively (2 out of 3)
    assert result == pytest.approx(2.0 / 3.0, abs=0.001)


@pytest.mark.paper
def test_substring_no_consecutive_match():
    """Test SUBSTRING when tokens don't appear consecutively."""
    claim_tokens = ["a", "b"]
    passage_tokens = ["a", "x", "b"]
    
    result = longest_common_substring_ratio(claim_tokens, passage_tokens)
    # "a" and "b" don't form a contiguous substring (only 1 match each)
    assert result == pytest.approx(0.5, abs=0.001)


@pytest.mark.paper
def test_substring_no_overlap():
    """Test SUBSTRING with no overlap."""
    claim_tokens = ["a", "b"]
    passage_tokens = ["x", "y"]
    
    result = longest_common_substring_ratio(claim_tokens, passage_tokens)
    assert result == pytest.approx(0.0, abs=0.001)


@pytest.mark.paper
def test_substring_empty_claim_raises():
    """Test that empty claim raises ValueError."""
    with pytest.raises(ValueError):
        longest_common_substring_ratio([], ["some", "tokens"])


# ============================================================================
# Integration: Real-World Examples
# ============================================================================

@pytest.mark.paper
def test_real_example_1():
    """Test on realistic claim/passage pair."""
    claim = tokenize("Paris is the capital of France")
    passage = tokenize("The capital of France is Paris")
    
    lco = longest_consecutive_overlap(claim, passage)
    lcs = longest_common_subsequence_ratio(claim, passage)
    substring = longest_common_substring_ratio(claim, passage)
    
    # All metrics should reflect high overlap (same words, different order)
    assert lco > 0.3  # At least "Paris" or "France" or "capital" or "is"
    assert lcs > 0.6  # Most words appear (just reordered)
    assert substring > 0.3


@pytest.mark.paper
def test_real_example_2():
    """Test on claim with moderate overlap."""
    claim = tokenize("The quick brown fox jumps")
    passage = tokenize("A quick brown fox runs fast")
    
    lco = longest_consecutive_overlap(claim, passage)
    lcs = longest_common_subsequence_ratio(claim, passage)
    substring = longest_common_substring_ratio(claim, passage)
    
    # Should have decent overlap but not perfect
    assert 0.3 < lco < 1.0
    assert 0.3 < lcs < 1.0
    assert 0.3 < substring < 1.0


@pytest.mark.paper
def test_real_example_3():
    """Test on minimal overlap."""
    claim = tokenize("Machine learning models are powerful")
    passage = tokenize("Cats are often playful animals")
    
    lco = longest_consecutive_overlap(claim, passage)
    lcs = longest_common_subsequence_ratio(claim, passage)
    substring = longest_common_substring_ratio(claim, passage)
    
    # Should have minimal overlap (only "are" in common)
    assert lco < 0.3
    assert lcs < 0.3
    assert substring < 0.3
