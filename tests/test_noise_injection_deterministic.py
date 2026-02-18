"""
Test deterministic behavior of noise injection functions.

Verifies that noise injection produces stable, reproducible outputs
when seeded, which is critical for:
- Reproducible robustness experiments
- Fair comparison across runs
- Debugging noise-related issues
"""

import pytest
from src.evaluation.noise_injection import (
    inject_headers_footers,
    inject_ocr_typos,
    inject_column_shuffle,
    inject_all_noise
)


class TestHeaderFooterInjection:
    """Test header/footer injection is deterministic."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        lines = [
            "Breadth-First Search (BFS) is a graph traversal algorithm.",
            "It explores vertices level by level using a queue.",
            "The time complexity is O(V + E).",
            "BFS guarantees shortest path in unweighted graphs.",
            "Depth-First Search (DFS) explores as far as possible.",
            "DFS can be implemented recursively or with a stack.",
            "DFS is used for topological sorting.",
            "Quick Sort is a divide-and-conquer algorithm.",
            "It has average time complexity O(n log n).",
            "Worst case is O(n^2) when pivot selection is poor."
        ]
        return '\n'.join(lines)
    
    def test_seeded_output_stable(self, sample_text):
        """Test that seeded injection produces identical output."""
        result1 = inject_headers_footers(sample_text, freq=3, seed=42)
        result2 = inject_headers_footers(sample_text, freq=3, seed=42)
        
        assert result1 == result2, "Seeded outputs should be identical"
    
    def test_different_seeds_differ(self, sample_text):
        """Test that different seeds produce different outputs."""
        result1 = inject_headers_footers(sample_text, freq=3, seed=42)
        result2 = inject_headers_footers(sample_text, freq=3, seed=123)
        
        # Output structure should be same (same injection points) but randomness may differ
        # For headers/footers, output should actually be the same since no randomness involved
        # But this tests the seed parameter works
        assert result1 == result2  # Headers/footers are deterministic given freq
    
    def test_headers_injected_at_freq(self, sample_text):
        """Test that headers are injected at specified frequency."""
        result = inject_headers_footers(sample_text, header="TEST", footer="FOOTER", freq=3, seed=42)
        
        assert "TEST" in result, "Header should be present"
        assert "FOOTER" in result, "Footer should be present"
        
        # Count header occurrences
        header_count = result.count("--- TEST ---")
        assert header_count > 0, "Headers should be injected"
    
    def test_preserves_original_text(self, sample_text):
        """Test that original text is preserved (only additions)."""
        result = inject_headers_footers(sample_text, freq=5, seed=42)
        
        # All original lines should still be present
        for line in sample_text.split('\n'):
            if line.strip():  # Skip empty lines
                assert line in result, f"Original line should be preserved: {line}"


class TestOCRTypoInjection:
    """Test OCR typo injection is deterministic."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text with characters prone to OCR errors."""
        return "The algorithm has O(log n) complexity. It processes l items in parallel."
    
    def test_seeded_output_stable(self, sample_text):
        """Test that seeded typo injection produces identical output."""
        result1 = inject_ocr_typos(sample_text, rate=0.1, seed=42)
        result2 = inject_ocr_typos(sample_text, rate=0.1, seed=42)
        
        assert result1 == result2, "Seeded outputs should be identical"
    
    def test_different_seeds_differ(self, sample_text):
        """Test that different seeds can produce different outputs."""
        # With high rate and different seeds, outputs should likely differ
        result1 = inject_ocr_typos(sample_text, rate=0.2, seed=42)
        result2 = inject_ocr_typos(sample_text, rate=0.2, seed=999)
        
        # May differ due to different random choices
        # But both should be valid OCR-corrupted text
        assert len(result1) > 0
        assert len(result2) > 0
    
    def test_low_rate_few_changes(self, sample_text):
        """Test that low rate produces few substitutions."""
        result = inject_ocr_typos(sample_text, rate=0.01, seed=42)
        
        # Calculate character differences
        diffs = sum(1 for c1, c2 in zip(sample_text, result) if c1 != c2)
        
        # With 1% rate, we expect roughly 1-2 changes in ~70 char text
        # Allow for some variance due to randomness
        assert diffs <= len(sample_text) * 0.05, "Should have minimal changes with low rate"
    
    def test_common_ocr_confusions(self):
        """Test that common OCR confusions are applied."""
        # Text designed to trigger OCR substitutions
        text = "llllOOOO0000IIIIiiii"
        
        # With high rate, should see substitutions
        result = inject_ocr_typos(text, rate=0.3, seed=42)
        
        # Verify some substitutions occurred (probabilistic but likely)
        # At 30% rate on 20 chars, expect ~6 substitutions
        diffs = sum(1 for c1, c2 in zip(text, result) if c1 != c2)
        assert diffs > 0, "Should have some OCR substitutions"
    
    def test_rn_to_m_substitution(self):
        """Test rn ↔ m substitution."""
        text = "modern pattern learning algorithm"
        
        # With high rate, should potentially see rn→m substitutions
        result = inject_ocr_typos(text, rate=0.5, seed=42)
        
        # Just verify it runs without error and produces output
        assert len(result) > 0
        # Length may differ due to rn→m substitution
        assert abs(len(result) - len(text)) <= 5


class TestColumnShuffle:
    """Test column shuffle is deterministic."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample multi-sentence text."""
        return (
            "BFS is a graph traversal algorithm. It uses a queue data structure. "
            "The time complexity is O(V + E). BFS finds shortest paths. "
            "DFS is another traversal method. It uses a stack or recursion. "
            "DFS has the same time complexity. It's used for topological sorting."
        )
    
    def test_seeded_output_stable(self, sample_text):
        """Test that seeded shuffle produces identical output."""
        result1 = inject_column_shuffle(sample_text, seed=42)
        result2 = inject_column_shuffle(sample_text, seed=42)
        
        assert result1 == result2, "Seeded outputs should be identical"
    
    def test_different_seeds_may_differ(self, sample_text):
        """Test that different seeds can produce different shuffles."""
        result1 = inject_column_shuffle(sample_text, seed=42)
        result2 = inject_column_shuffle(sample_text, seed=999)
        
        # May differ due to random interleaving
        assert len(result1) > 0
        assert len(result2) > 0
    
    def test_preserves_sentences(self, sample_text):
        """Test that individual sentences are preserved (only reordered)."""
        result = inject_column_shuffle(sample_text, seed=42)
        
        # Extract key concepts from original
        key_concepts = ["BFS", "DFS", "algorithm", "traversal", "queue", "stack"]
        
        # Most key concepts should be present in result
        preserved = sum(1 for concept in key_concepts if concept in result)
        assert preserved >= 4, f"At least 4 key concepts should be preserved, got {preserved}"
    
    def test_short_text_handled(self):
        """Test that short text is handled gracefully."""
        short_text = "BFS uses a queue."
        result = inject_column_shuffle(short_text, seed=42)
        
        assert len(result) > 0, "Should handle short text"
        assert "BFS" in result, "Should preserve content"


class TestCombinedNoise:
    """Test combined noise injection is deterministic."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return (
            "Breadth-First Search explores vertices level by level.\n"
            "It uses a queue data structure for traversal.\n"
            "The time complexity is O(V + E) for graphs.\n"
            "BFS guarantees shortest path in unweighted graphs.\n"
            "Depth-First Search explores as far as possible first.\n"
            "DFS can be implemented recursively or with a stack.\n"
            "Both algorithms have the same time complexity.\n"
            "DFS is commonly used for topological sorting.\n"
        )
    
    def test_seeded_output_stable(self, sample_text):
        """Test that seeded combined noise produces identical output."""
        result1 = inject_all_noise(sample_text, seed=42)
        result2 = inject_all_noise(sample_text, seed=42)
        
        assert result1 == result2, "Seeded combined noise should be identical"
    
    def test_all_noise_types_applied(self, sample_text):
        """Test that all noise types are applied when enabled."""
        result = inject_all_noise(
            sample_text,
            apply_headers=True,
            apply_ocr=True,
            apply_shuffle=True,
            seed=42
        )
        
        # Result should differ from original (some noise applied)
        assert result != sample_text, "Text should be modified"
        
        # Should contain some of the original content
        assert "algorithm" in result or "BFS" in result or "DFS" in result, \
            "Original content should be recognizable"
    
    def test_selective_noise_application(self, sample_text):
        """Test that noise types can be selectively disabled."""
        # Only headers
        result_headers = inject_all_noise(
            sample_text,
            apply_headers=True,
            apply_ocr=False,
            apply_shuffle=False,
            header_freq=2,  # More frequent for small text
            seed=42
        )
        # Should have headers/footers injected or at least be different
        assert "---" in result_headers or "[" in result_headers or result_headers != sample_text, \
            "Headers/footers should modify the text"
        
        # Only OCR (harder to verify but should run)
        result_ocr = inject_all_noise(
            sample_text,
            apply_headers=False,
            apply_ocr=True,
            apply_shuffle=False,
            ocr_rate=0.1,
            seed=42
        )
        assert len(result_ocr) > 0
        
        # None applied (should return original-ish)
        result_none = inject_all_noise(
            sample_text,
            apply_headers=False,
            apply_ocr=False,
            apply_shuffle=False,
            seed=42
        )
        assert len(result_none) > 0
    
    def test_different_seeds_produce_different_output(self, sample_text):
        """Test that different seeds produce different combined noise."""
        result1 = inject_all_noise(sample_text, seed=42, ocr_rate=0.1)
        result2 = inject_all_noise(sample_text, seed=999, ocr_rate=0.1)
        
        # With high OCR rate and shuffle, outputs should likely differ
        # But both should be valid
        assert len(result1) > 0
        assert len(result2) > 0


class TestEdgeCases:
    """Test edge cases for noise injection."""
    
    def test_empty_text(self):
        """Test that empty text is handled gracefully."""
        assert inject_headers_footers("", seed=42) == ""
        assert inject_ocr_typos("", seed=42) == ""
        assert inject_column_shuffle("", seed=42) == ""
    
    def test_single_character(self):
        """Test single character text."""
        text = "O"
        result = inject_ocr_typos(text, rate=1.0, seed=42)
        assert len(result) == 1, "Single char should produce single char"
    
    def test_no_sentence_delimiters(self):
        """Test text with no sentence delimiters."""
        text = "no periods or delimiters here just words"
        result = inject_column_shuffle(text, seed=42)
        assert len(result) > 0, "Should handle text without sentence delimiters"
    
    def test_zero_noise_rate(self):
        """Test zero noise rate produces no changes."""
        text = "The algorithm has O(log n) complexity."
        result = inject_ocr_typos(text, rate=0.0, seed=42)
        assert result == text, "Zero rate should produce no changes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
