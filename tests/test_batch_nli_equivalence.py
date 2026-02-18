"""
Tests for batch NLI verification equivalence.

Validates that batch processing produces equivalent results to single-call processing
within acceptable floating-point tolerance.

Constraints:
- Batch and single results must be within 1e-5 tolerance
- Label predictions must be identical
- Order must be preserved
"""

import pytest
import numpy as np
from typing import List, Tuple

from src.claims.nli_verifier import NLIVerifier, EntailmentLabel, NLIResult


class TestBatchNLIEquivalence:
    """Test batch NLI processing produces equivalent results to single calls."""
    
    @pytest.fixture
    def verifier(self):
        """Provide NLI verifier instance."""
        # Use small batch size for testing
        return NLIVerifier(
            model_name="roberta-large-mnli",
            device="cpu",
            batch_size=4
        )
    
    @pytest.fixture
    def test_pairs(self) -> List[Tuple[str, str]]:
        """Provide diverse test pairs."""
        return [
            ("The sky is blue.", "Blue is a color."),
            ("Dogs are mammals.", "Cats are reptiles."),
            ("Paris is the capital of France.", "France is in Europe."),
            ("Water boils at 100 degrees Celsius.", "Ice melts at 0 degrees Celsius."),
            ("Machine learning uses algorithms.", "Algorithms process data."),
            ("The Earth orbits the Sun.", "The Sun is a star."),
            ("COVID-19 is a virus.", "Viruses are living organisms."),
            ("Python is a programming language.", "Programming languages are tools for coding."),
        ]
    
    def test_single_vs_batch_equivalence(self, verifier, test_pairs):
        """Test that batch processing produces same results as single calls."""
        
        # Process individually
        single_results = []
        for claim, evidence in test_pairs:
            result = verifier.verify(claim, evidence)
            single_results.append(result)
        
        # Process in batch
        batch_results = verifier.verify_batch(test_pairs)
        
        # Compare
        assert len(batch_results) == len(single_results), "Batch result count mismatch"
        
        for i, (single, batch) in enumerate(zip(single_results, batch_results)):
            # Compare labels (must be identical)
            assert single.label == batch.label, \
                f"Label mismatch at index {i}: {single.label} vs {batch.label}"
            
            # Compare probabilities (within tolerance)
            tolerance = 1e-5
            assert abs(single.entailment_prob - batch.entailment_prob) < tolerance, \
                f"Entailment prob mismatch at {i}: {single.entailment_prob} vs {batch.entailment_prob}"
            assert abs(single.contradiction_prob - batch.contradiction_prob) < tolerance, \
                f"Contradiction prob mismatch at {i}: {single.contradiction_prob} vs {batch.contradiction_prob}"
            assert abs(single.neutral_prob - batch.neutral_prob) < tolerance, \
                f"Neutral prob mismatch at {i}: {single.neutral_prob} vs {batch.neutral_prob}"
    
    def test_batch_order_preserved(self, verifier, test_pairs):
        """Test that batch processing preserves order of input pairs."""
        
        batch_results = verifier.verify_batch(test_pairs)
        
        # Verify order is preserved by comparing with individual processing
        for i, (claim, evidence) in enumerate(test_pairs):
            single_result = verifier.verify(claim, evidence)
            batch_result = batch_results[i]
            
            assert single_result.label == batch_result.label
            assert abs(single_result.entailment_prob - batch_result.entailment_prob) < 1e-5
    
    def test_batch_with_scores_equivalence(self, verifier, test_pairs):
        """Test batch processing with scores output."""
        
        results, scores = verifier.verify_batch_with_scores(test_pairs)
        
        # Verify scores shape
        assert scores.shape == (len(test_pairs), 3)
        assert scores.dtype == np.float32
        
        # Verify scores match results
        for i, result in enumerate(results):
            assert abs(scores[i, 0] - result.entailment_prob) < 1e-5
            assert abs(scores[i, 1] - result.contradiction_prob) < 1e-5
            assert abs(scores[i, 2] - result.neutral_prob) < 1e-5
    
    def test_empty_batch(self, verifier):
        """Test handling of empty batch."""
        
        results = verifier.verify_batch([])
        assert results == []
        
        results, scores = verifier.verify_batch_with_scores([])
        assert results == []
        assert scores.shape == (0, 3)
    
    def test_batch_with_different_batch_sizes(self, test_pairs):
        """Test that different batch sizes produce same results."""
        
        batch_sizes = [1, 2, 4, 8, 16]
        reference_results = None
        
        for batch_size in batch_sizes:
            verifier = NLIVerifier(batch_size=batch_size, device="cpu")
            results = verifier.verify_batch(test_pairs)
            
            if reference_results is None:
                reference_results = results
            else:
                # Compare with reference
                assert len(results) == len(reference_results)
                for i, (r, ref) in enumerate(zip(results, reference_results)):
                    assert r.label == ref.label
                    assert abs(r.entailment_prob - ref.entailment_prob) < 1e-5
                    assert abs(r.contradiction_prob - ref.contradiction_prob) < 1e-5
                    assert abs(r.neutral_prob - ref.neutral_prob) < 1e-5
    
    def test_large_batch(self, verifier):
        """Test processing of large batch."""
        
        # Generate 100 diverse pairs
        pairs = []
        for i in range(100):
            pairs.append((
                f"Statement {i}: This is a test claim about topic {i%10}.",
                f"Supporting evidence {i}: This discusses topic {i%10} in detail."
            ))
        
        # Process batch
        results = verifier.verify_batch(pairs)
        
        # Verify results
        assert len(results) == len(pairs)
        for result in results:
            assert isinstance(result, NLIResult)
            assert 0 <= result.entailment_prob <= 1
            assert 0 <= result.contradiction_prob <= 1
            assert 0 <= result.neutral_prob <= 1
            # Probabilities should sum to ~1.0
            assert abs(sum([result.entailment_prob, result.contradiction_prob, result.neutral_prob]) - 1.0) < 1e-5
    
    def test_confidence_computation(self, verifier, test_pairs):
        """Test confidence property computation."""
        
        results = verifier.verify_batch(test_pairs)
        
        for result in results:
            expected_confidence = max(
                result.entailment_prob,
                result.contradiction_prob,
                result.neutral_prob
            )
            assert abs(result.confidence - expected_confidence) < 1e-10
    
    def test_entailment_detection(self, verifier):
        """Test that verifier correctly detects entailment."""
        
        # Strong entailment pair
        claim = "The capital of France is Paris"
        evidence = "Paris is the capital of France"
        
        result = verifier.verify(claim, evidence)
        assert result.label == EntailmentLabel.ENTAILMENT
        assert result.entailment_prob > result.contradiction_prob
        assert result.entailment_prob > result.neutral_prob
    
    def test_contradiction_detection(self, verifier):
        """Test that verifier detects contradictions."""
        
        # Contradictory pair
        claim = "The Earth is flat"
        evidence = "The Earth is a sphere"
        
        result = verifier.verify(claim, evidence)
        assert result.label == EntailmentLabel.CONTRADICTION
        assert result.contradiction_prob > result.entailment_prob
    
    def test_neutral_detection(self, verifier):
        """Test that verifier detects neutral cases."""
        
        # Neutral pair (no logical relationship)
        claim = "The color red is bright"
        evidence = "Cats can see in the dark"
        
        result = verifier.verify(claim, evidence)
        assert result.label == EntailmentLabel.NEUTRAL
    
    def test_deterministic_output(self, verifier, test_pairs):
        """Test that same input produces deterministic output."""
        
        results1 = verifier.verify_batch(test_pairs)
        results2 = verifier.verify_batch(test_pairs)
        
        for r1, r2 in zip(results1, results2):
            # Should be exactly identical (to floating point precision)
            assert r1.label == r2.label
            assert r1.entailment_prob == r2.entailment_prob
            assert r1.contradiction_prob == r2.contradiction_prob
            assert r1.neutral_prob == r2.neutral_prob
    
    def test_consensus_batch_equivalence(self, verifier):
        """Test consensus checking with batch equivalence."""
        
        claim = "Machine learning is a subset of AI"
        evidence_list = [
            "Artificial intelligence is a broad field",
            "Machine learning uses algorithms",
            "AI includes machine learning and other techniques",
        ]
        
        # Check consensus
        consensus = verifier.check_consensus(claim, evidence_list)
        
        # Verify against batch processing
        pairs = [(claim, ev) for ev in evidence_list]
        batch_results = verifier.verify_batch(pairs)
        
        # Compare
        assert len(consensus["results"]) == len(batch_results)
        for i, (cons_result, batch_result) in enumerate(zip(consensus["results"], batch_results)):
            assert cons_result.label == batch_result.label


class TestBatchPerformance:
    """Test performance characteristics of batch processing."""
    
    def test_batch_faster_than_sequential(self):
        """Test that batch processing is faster than sequential."""
        
        import time
        
        verifier = NLIVerifier(batch_size=16, device="cpu")
        
        # Generate test pairs
        pairs = [
            (f"Claim {i}", f"Evidence {i}")
            for i in range(20)
        ]
        
        # Time sequential processing
        start = time.time()
        for claim, evidence in pairs:
            verifier.verify(claim, evidence)
        sequential_time = time.time() - start
        
        # Time batch processing
        start = time.time()
        verifier.verify_batch(pairs)
        batch_time = time.time() - start
        
        # Batch should be faster (or at least not significantly slower)
        # Account for overhead
        assert batch_time <= sequential_time * 1.5


class TestBatchEdgeCases:
    """Test edge cases in batch processing."""
    
    def test_very_long_text(self):
        """Test handling of very long claims and evidence."""
        
        verifier = NLIVerifier(batch_size=4, device="cpu")
        
        # Very long text (should be truncated to 512 tokens)
        long_text = " ".join(["word"] * 1000)
        
        pair = (long_text, long_text)
        result = verifier.verify(*pair)
        
        assert isinstance(result, NLIResult)
        assert 0 <= result.entailment_prob <= 1
    
    def test_special_characters(self):
        """Test handling of special characters."""
        
        verifier = NLIVerifier(device="cpu")
        
        pairs = [
            ("Hello, world!", "Hi there!"),
            ("What's up?", "How are you?"),
            ("Email: test@example.com", "Contact information included"),
            ("Price: $99.99", "Cost of product is ninety-nine dollars"),
        ]
        
        results = verifier.verify_batch(pairs)
        assert len(results) == len(pairs)
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        
        verifier = NLIVerifier(device="cpu")
        
        pairs = [
            ("Bonjour, ça va?", "Hello, how are you?"),
            ("北京是中国的首都", "Beijing is the capital of China"),
            ("Привет, как дела?", "Hi, how are you?"),
        ]
        
        results = verifier.verify_batch(pairs)
        assert len(results) == len(pairs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
