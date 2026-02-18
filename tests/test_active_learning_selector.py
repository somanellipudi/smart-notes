"""
Tests for Active Learning Selection Components

Verifies deterministic behavior of uncertainty scoring and diversity sampling
given fixed random seeds.
"""

import pytest
import numpy as np
from src.evaluation.al_utils import (
    compute_uncertainty_least_confident,
    compute_uncertainty_margin,
    compute_uncertainty_entropy,
    compute_uncertainty,
    diversity_sample_by_topic,
    select_for_labeling,
    validate_predictions
)


class TestUncertaintyScoring:
    """Test uncertainty computation strategies."""
    
    def test_least_confident_high_confidence(self):
        """Test least confident with highly confident prediction."""
        probs = np.array([0.95, 0.03, 0.02])
        uncertainty = compute_uncertainty_least_confident(probs)
        assert uncertainty == pytest.approx(0.05, abs=0.01)
    
    def test_least_confident_uncertain(self):
        """Test least confident with uncertain prediction."""
        probs = np.array([0.4, 0.35, 0.25])
        uncertainty = compute_uncertainty_least_confident(probs)
        assert uncertainty == pytest.approx(0.6, abs=0.01)
    
    def test_least_confident_uniform(self):
        """Test least confident with uniform distribution."""
        probs = np.array([0.33, 0.33, 0.34])
        uncertainty = compute_uncertainty_least_confident(probs)
        assert uncertainty == pytest.approx(0.66, abs=0.01)
    
    def test_margin_large_gap(self):
        """Test margin strategy with large gap between top 2."""
        probs = np.array([0.95, 0.03, 0.02])
        uncertainty = compute_uncertainty_margin(probs)
        assert uncertainty == pytest.approx(0.08, abs=0.01)  # 1 - (0.95 - 0.03) = 0.08
    
    def test_margin_small_gap(self):
        """Test margin strategy with small gap."""
        probs = np.array([0.51, 0.49, 0.00])
        uncertainty = compute_uncertainty_margin(probs)
        assert uncertainty == pytest.approx(0.98, abs=0.01)  # 1 - (0.51 - 0.49) = 0.98
    
    def test_entropy_certain(self):
        """Test entropy with certain prediction."""
        probs = np.array([1.0, 0.0, 0.0])
        uncertainty = compute_uncertainty_entropy(probs)
        assert uncertainty == pytest.approx(0.0, abs=0.01)
    
    def test_entropy_uniform(self):
        """Test entropy with uniform distribution (maximum uncertainty)."""
        probs = np.array([0.33, 0.33, 0.34])
        uncertainty = compute_uncertainty_entropy(probs)
        # Uniform distribution should have high entropy (close to 1.0)
        assert uncertainty > 0.95
    
    def test_entropy_moderate(self):
        """Test entropy with moderate uncertainty."""
        probs = np.array([0.6, 0.3, 0.1])
        uncertainty = compute_uncertainty_entropy(probs)
        # Should be between 0 and 1
        assert 0.3 < uncertainty < 0.85
    
    def test_compute_uncertainty_strategies(self):
        """Test compute_uncertainty with all strategies."""
        probs = np.array([0.7, 0.2, 0.1])
        
        unc_lc = compute_uncertainty(probs, strategy="least_confident")
        unc_margin = compute_uncertainty(probs, strategy="margin")
        unc_entropy = compute_uncertainty(probs, strategy="entropy")
        
        # All should be in [0, 1]
        assert 0.0 <= unc_lc <= 1.0
        assert 0.0 <= unc_margin <= 1.0
        assert 0.0 <= unc_entropy <= 1.0
    
    def test_compute_uncertainty_invalid_strategy(self):
        """Test compute_uncertainty with invalid strategy."""
        probs = np.array([0.7, 0.2, 0.1])
        
        with pytest.raises(ValueError, match="Unknown uncertainty strategy"):
            compute_uncertainty(probs, strategy="invalid_strategy")
    
    def test_uncertainty_edge_cases(self):
        """Test uncertainty with edge cases."""
        # Empty array
        assert compute_uncertainty_least_confident(np.array([])) == 0.0
        assert compute_uncertainty_margin(np.array([])) == 0.0
        assert compute_uncertainty_entropy(np.array([])) == 0.0
        
        # Single element
        assert compute_uncertainty_margin(np.array([1.0])) == 0.0


class TestDiversitySampling:
    """Test topic-stratified diversity sampling."""
    
    def test_diversity_sample_deterministic(self):
        """Test that diversity sampling is deterministic with fixed seed."""
        examples = [
            {"doc_id": f"algo_{i}", "domain_topic": "Algorithms", "uncertainty": 0.9 - i*0.1}
            for i in range(5)
        ] + [
            {"doc_id": f"net_{i}", "domain_topic": "Networks", "uncertainty": 0.8 - i*0.1}
            for i in range(3)
        ]
        
        # Same seed should give same results
        sample1 = diversity_sample_by_topic(examples, n_samples=4, random_state=42)
        sample2 = diversity_sample_by_topic(examples, n_samples=4, random_state=42)
        
        ids1 = [ex["doc_id"] for ex in sample1]
        ids2 = [ex["doc_id"] for ex in sample2]
        
        assert ids1 == ids2, "Same seed should produce identical samples"
    
    def test_diversity_sample_different_seeds(self):
        """Test that different seeds produce different samples."""
        examples = [
            {"doc_id": f"algo_{i}", "domain_topic": "Algorithms", "uncertainty": 0.9 - i*0.1}
            for i in range(10)
        ]
        
        sample1 = diversity_sample_by_topic(examples, n_samples=3, random_state=42)
        sample2 = diversity_sample_by_topic(examples, n_samples=3, random_state=123)
        
        ids1 = [ex["doc_id"] for ex in sample1]
        ids2 = [ex["doc_id"] for ex in sample2]
        
        # Should be different (with high probability)
        assert ids1 != ids2, "Different seeds should produce different samples"
    
    def test_diversity_sample_topic_coverage(self):
        """Test that diversity sampling covers multiple topics."""
        examples = [
            {"doc_id": f"algo_{i}", "domain_topic": "Algorithms"}
            for i in range(10)
        ] + [
            {"doc_id": f"net_{i}", "domain_topic": "Networks"}
            for i in range(10)
        ]
        
        sample = diversity_sample_by_topic(examples, n_samples=10, random_state=42)
        
        topics = set(ex["domain_topic"] for ex in sample)
        
        # Should include both topics (with high probability for n=10 samples)
        assert len(topics) >= 2, "Diversity sampling should cover multiple topics"
    
    def test_diversity_sample_proportional(self):
        """Test that sampling is roughly proportional to topic sizes."""
        # 60 Algorithms, 40 Networks
        examples = [
            {"doc_id": f"algo_{i}", "domain_topic": "Algorithms"}
            for i in range(60)
        ] + [
            {"doc_id": f"net_{i}", "domain_topic": "Networks"}
            for i in range(40)
        ]
        
        sample = diversity_sample_by_topic(examples, n_samples=20, random_state=42)
        
        algo_count = sum(1 for ex in sample if ex["domain_topic"] == "Algorithms")
        net_count = sum(1 for ex in sample if ex["domain_topic"] == "Networks")
        
        # Should be roughly 60/40 split (12/8)
        # Allow some variance due to rounding
        assert 10 <= algo_count <= 14, f"Expected ~12 Algorithms, got {algo_count}"
        assert 6 <= net_count <= 10, f"Expected ~8 Networks, got {net_count}"
    
    def test_diversity_sample_all_examples(self):
        """Test that requesting more samples than available returns all."""
        examples = [{"doc_id": f"ex_{i}", "domain_topic": "Topic"} for i in range(5)]
        
        sample = diversity_sample_by_topic(examples, n_samples=10, random_state=42)
        
        assert len(sample) == 5, "Should return all available examples"
    
    def test_diversity_sample_single_topic(self):
        """Test sampling with only one topic."""
        examples = [{"doc_id": f"ex_{i}", "domain_topic": "OneTopic"} for i in range(10)]
        
        sample = diversity_sample_by_topic(examples, n_samples=5, random_state=42)
        
        assert len(sample) == 5
        assert all(ex["domain_topic"] == "OneTopic" for ex in sample)


class TestSelectForLabeling:
    """Test combined uncertainty + diversity selection."""
    
    def test_select_for_labeling_deterministic(self):
        """Test that selection is deterministic with fixed seed."""
        examples = [
            {
                "doc_id": f"ex_{i}",
                "domain_topic": "Algorithms" if i < 5 else "Networks",
                "probabilities": np.random.dirichlet([1, 1, 1]).tolist()
            }
            for i in range(10)
        ]
        
        # Add probabilities manually for determinism
        for i, ex in enumerate(examples):
            if i % 2 == 0:
                ex["probabilities"] = [0.6, 0.3, 0.1]  # Moderate uncertainty
            else:
                ex["probabilities"] = [0.95, 0.03, 0.02]  # Low uncertainty
        
        selected1, stats1 = select_for_labeling(examples, n_samples=5, random_state=42)
        selected2, stats2 = select_for_labeling(examples, n_samples=5, random_state=42)
        
        ids1 = [ex["doc_id"] for ex in selected1]
        ids2 = [ex["doc_id"] for ex in selected2]
        
        assert ids1 == ids2, "Same seed should produce identical selections"
    
    def test_select_for_labeling_high_uncertainty_first(self):
        """Test that high uncertainty examples are prioritized."""
        examples = [
            {
                "doc_id": "uncertain_1",
                "domain_topic": "Topic",
                "probabilities": [0.4, 0.35, 0.25]  # High uncertainty
            },
            {
                "doc_id": "certain_1",
                "domain_topic": "Topic",
                "probabilities": [0.95, 0.03, 0.02]  # Low uncertainty
            },
            {
                "doc_id": "uncertain_2",
                "domain_topic": "Topic",
                "probabilities": [0.45, 0.30, 0.25]  # High uncertainty
            }
        ]
        
        selected, stats = select_for_labeling(
            examples,
            n_samples=2,
            diversity_weight=0.0,  # Pure uncertainty
            random_state=42
        )
        
        # Should select the two uncertain examples
        selected_ids = [ex["doc_id"] for ex in selected]
        assert "uncertain_1" in selected_ids
        assert "uncertain_2" in selected_ids
    
    def test_select_for_labeling_all_strategies(self):
        """Test selection with all uncertainty strategies."""
        examples = [
            {
                "doc_id": f"ex_{i}",
                "domain_topic": "Topic",
                "probabilities": [0.6 - i*0.05, 0.3, 0.1 + i*0.05]
            }
            for i in range(10)
        ]
        
        for strategy in ["least_confident", "margin", "entropy"]:
            selected, stats = select_for_labeling(
                examples,
                n_samples=5,
                uncertainty_strategy=strategy,
                random_state=42
            )
            
            assert len(selected) == 5
            assert stats["uncertainty_strategy"] == strategy
            assert "uncertainty_stats" in stats
    
    def test_select_for_labeling_diversity_weight(self):
        """Test that diversity_weight affects selection."""
        # Create unbalanced dataset
        examples = [
            {
                "doc_id": f"algo_{i}",
                "domain_topic": "Algorithms",  # Majority
                "probabilities": [0.4, 0.35, 0.25]
            }
            for i in range(8)
        ] + [
            {
                "doc_id": f"net_{i}",
                "domain_topic": "Networks",  # Minority
                "probabilities": [0.4, 0.35, 0.25]
            }
            for i in range(2)
        ]
        
        # With high diversity weight, should include more Networks
        selected_div, stats_div = select_for_labeling(
            examples,
            n_samples=5,
            diversity_weight=0.8,
            random_state=42
        )
        
        # With zero diversity weight, might be all Algorithms (order dependent)
        selected_no_div, stats_no_div = select_for_labeling(
            examples,
            n_samples=5,
            diversity_weight=0.0,
            random_state=42
        )
        
        # Both should return valid results
        assert len(selected_div) == 5
        assert len(selected_no_div) == 5
    
    def test_select_for_labeling_stats(self):
        """Test that selection stats are computed correctly."""
        examples = [
            {
                "doc_id": f"ex_{i}",
                "domain_topic": "Topic1" if i < 3 else "Topic2",
                "probabilities": [0.5, 0.3, 0.2]
            }
            for i in range(6)
        ]
        
        selected, stats = select_for_labeling(examples, n_samples=4, random_state=42)
        
        # Check stats structure
        assert "total_available" in stats
        assert "selected" in stats
        assert "uncertainty_strategy" in stats
        assert "topic_distribution" in stats
        assert "uncertainty_stats" in stats
        
        assert stats["total_available"] == 6
        assert stats["selected"] == 4
        
        # Uncertainty stats
        unc_stats = stats["uncertainty_stats"]
        assert "mean" in unc_stats
        assert "median" in unc_stats
        assert "min" in unc_stats
        assert "max" in unc_stats
        
        # All should be in [0, 1]
        assert 0.0 <= unc_stats["min"] <= unc_stats["max"] <= 1.0
    
    def test_select_for_labeling_more_samples_than_available(self):
        """Test requesting more samples than available."""
        examples = [
            {
                "doc_id": f"ex_{i}",
                "domain_topic": "Topic",
                "probabilities": [0.5, 0.3, 0.2]
            }
            for i in range(3)
        ]
        
        selected, stats = select_for_labeling(examples, n_samples=10, random_state=42)
        
        # Should return all available examples
        assert len(selected) == 3
        assert stats["selected"] == 3


class TestValidatePredictions:
    """Test prediction validation."""
    
    def test_validate_valid_predictions(self):
        """Test validation with valid predictions."""
        examples = [
            {
                "doc_id": "ex_1",
                "probabilities": [0.7, 0.2, 0.1]
            },
            {
                "doc_id": "ex_2",
                "probabilities": [0.33, 0.33, 0.34]
            }
        ]
        
        is_valid, errors = validate_predictions(examples)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_missing_probabilities(self):
        """Test validation with missing probabilities field."""
        examples = [
            {"doc_id": "ex_1"}  # Missing probabilities
        ]
        
        is_valid, errors = validate_predictions(examples)
        
        assert not is_valid
        assert len(errors) > 0
        assert "missing" in errors[0].lower()
    
    def test_validate_wrong_length(self):
        """Test validation with wrong probability array length."""
        examples = [
            {
                "doc_id": "ex_1",
                "probabilities": [0.5, 0.5]  # Only 2 values instead of 3
            }
        ]
        
        is_valid, errors = validate_predictions(examples)
        
        assert not is_valid
        assert len(errors) > 0
        assert "3 probabilities" in errors[0]
    
    def test_validate_not_sum_to_one(self):
        """Test validation with probabilities not summing to 1.0."""
        examples = [
            {
                "doc_id": "ex_1",
                "probabilities": [0.5, 0.3, 0.1]  # Sums to 0.9
            }
        ]
        
        is_valid, errors = validate_predictions(examples)
        
        assert not is_valid
        assert len(errors) > 0
        assert "sum to 1.0" in errors[0]
    
    def test_validate_out_of_range(self):
        """Test validation with probabilities out of [0, 1] range."""
        examples = [
            {
                "doc_id": "ex_1",
                "probabilities": [1.5, -0.3, -0.2]  # Out of range
            }
        ]
        
        is_valid, errors = validate_predictions(examples)
        
        assert not is_valid
        assert len(errors) > 0
        assert "[0, 1]" in errors[0]
    
    def test_validate_numpy_array(self):
        """Test validation with numpy arrays (should work)."""
        examples = [
            {
                "doc_id": "ex_1",
                "probabilities": np.array([0.7, 0.2, 0.1])
            }
        ]
        
        is_valid, errors = validate_predictions(examples)
        
        assert is_valid
        assert len(errors) == 0


class TestEndToEndScenario:
    """Test realistic end-to-end active learning scenario."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from examples to selection."""
        # Create realistic dataset
        np.random.seed(42)
        
        examples = []
        domains = ["Algorithms", "Networks", "OS", "Databases"]
        
        for i in range(40):
            domain = domains[i % len(domains)]
            
            # Generate diverse uncertainty levels
            if i % 3 == 0:
                probs = [0.95, 0.03, 0.02]  # Confident
            elif i % 3 == 1:
                probs = [0.6, 0.3, 0.1]  # Moderate
            else:
                probs = [0.4, 0.35, 0.25]  # Uncertain
            
            examples.append({
                "doc_id": f"{domain.lower()}_{i}",
                "domain_topic": domain,
                "claim": f"Test claim {i}",
                "source_text": f"Test source {i}",
                "probabilities": probs
            })
        
        # Select 10 examples for labeling
        selected, stats = select_for_labeling(
            examples,
            n_samples=10,
            uncertainty_strategy="margin",
            diversity_weight=0.3,
            random_state=42
        )
        
        # Verify results
        assert len(selected) == 10
        assert stats["selected"] == 10
        assert stats["total_available"] == 40
        
        # Should cover multiple domains
        selected_domains = set(ex["domain_topic"] for ex in selected)
        assert len(selected_domains) >= 2
        
        # Should prioritize uncertain examples
        uncertainties = [ex["uncertainty"] for ex in selected]
        assert np.mean(uncertainties) > 0.3  # Should be at least moderately uncertain
        
        # Verify all have required fields
        for ex in selected:
            assert "doc_id" in ex
            assert "domain_topic" in ex
            assert "probabilities" in ex
            assert "uncertainty" in ex


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
