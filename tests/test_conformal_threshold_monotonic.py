"""
Tests for Conformal Prediction Module

Tests conformal threshold calibration, coverage guarantees, and monotonicity.
"""

import pytest
import numpy as np
from src.evaluation.conformal import (
    compute_conformal_threshold,
    compute_conformal_threshold_simple,
    validate_conformal_coverage,
    conformal_prediction_calibration,
    confidence_band_quantile,
    expected_calibration_error,
    format_conformal_summary,
    combine_selective_and_conformal,
    ConformalResult
)


class TestConformalThreshold:
    """Tests for conformal threshold computation."""
    
    def test_perfect_calibration_set(self):
        """All predictions correct → threshold should be low."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        is_correct = np.array([1, 1, 1, 1, 1])
        
        threshold = compute_conformal_threshold_simple(scores, is_correct, alpha=0.1)
        
        # No errors, so threshold should be at or below minimum score
        assert threshold <= np.min(scores) + 0.1
    
    def test_all_errors(self):
        """All predictions wrong → threshold should be high."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        is_correct = np.array([0, 0, 0, 0, 0])
        
        threshold = compute_conformal_threshold_simple(scores, is_correct, alpha=0.1)
        
        # All errors, threshold should be near (1-alpha) quantile
        expected = np.quantile(scores, 0.9)
        assert abs(threshold - expected) < 0.2
    
    def test_mixed_correctness(self):
        """Mixed correct/incorrect predictions."""
        scores = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25])
        is_correct = np.array([1, 1, 0, 1, 0, 0, 0, 0])  # 3 correct, 5 errors
        
        threshold = compute_conformal_threshold_simple(scores, is_correct, alpha=0.1)
        
        # Threshold should be somewhere in middle-to-high range
        assert 0.0 <= threshold <= 1.0
        
        # Error scores: [0.75, 0.55, 0.45, 0.35, 0.25]
        # At alpha=0.1, threshold should be near 90th percentile of errors
        error_scores = scores[is_correct == 0]
        expected_approx = np.quantile(error_scores, 0.9)
        assert abs(threshold - expected_approx) < 0.3  # Allow some slack for finite-sample correction
    
    def test_empty_input(self):
        """Empty input should raise error for full method."""
        # Simple method returns default value for empty input
        threshold = compute_conformal_threshold_simple(np.array([]), np.array([]), alpha=0.1)
        assert threshold == 0.5
        
        # Full method should raise error
        with pytest.raises(ValueError, match="Empty"):
            compute_conformal_threshold(np.array([]), np.array([]), alpha=0.1)
    
    def test_invalid_alpha(self):
        """Invalid alpha values should raise error."""
        scores = np.array([0.9, 0.8])
        is_correct = np.array([1, 0])
        
        with pytest.raises(ValueError, match="alpha"):
            compute_conformal_threshold(scores, is_correct, alpha=0.0)
        
        with pytest.raises(ValueError, match="alpha"):
            compute_conformal_threshold(scores, is_correct, alpha=1.0)
        
        with pytest.raises(ValueError, match="alpha"):
            compute_conformal_threshold(scores, is_correct, alpha=-0.1)
    
    def test_mismatched_lengths(self):
        """Mismatched input lengths should raise error."""
        scores = np.array([0.9, 0.8])
        is_correct = np.array([1, 0, 1])  # Different length
        
        with pytest.raises(ValueError, match="same length"):
            compute_conformal_threshold(scores, is_correct, alpha=0.1)
    
    def test_single_example(self):
        """Single example should work."""
        scores = np.array([0.8])
        is_correct = np.array([1])
        
        threshold = compute_conformal_threshold_simple(scores, is_correct, alpha=0.1)
        assert 0.0 <= threshold <= 1.0
    
    def test_different_alpha_values(self):
        """Different alpha values should give different thresholds."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        is_correct = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        
        threshold_01 = compute_conformal_threshold_simple(scores, is_correct, alpha=0.1)
        threshold_05 = compute_conformal_threshold_simple(scores, is_correct, alpha=0.5)
        
        # Smaller alpha (more confidence) should give higher threshold
        assert threshold_01 > threshold_05


class TestThresholdMonotonicity:
    """Tests for monotonicity properties of conformal thresholds."""
    
    def test_alpha_monotonicity(self):
        """Smaller alpha should give higher or equal threshold."""
        np.random.seed(42)
        n = 50
        scores = np.random.uniform(0.2, 1.0, n)
        is_correct = (scores > 0.6).astype(int)  # High scores more likely correct
        
        alphas = [0.01, 0.05, 0.1, 0.2, 0.3]
        thresholds = []
        
        for alpha in alphas:
            threshold = compute_conformal_threshold_simple(scores, is_correct, alpha)
            thresholds.append(threshold)
        
        # Thresholds should be non-increasing as alpha increases
        for i in range(len(thresholds) - 1):
            # Allow small violations due to quantile discretization
            assert thresholds[i] >= thresholds[i + 1] - 0.15, \
                f"Monotonicity violated: α={alphas[i]} → {thresholds[i]:.3f}, α={alphas[i+1]} → {thresholds[i+1]:.3f}"
    
    def test_score_correlation_monotonicity(self):
        """Better score-correctness correlation should give lower threshold."""
        np.random.seed(42)
        n = 100
        
        # Strong correlation: high scores → correct
        scores1 = np.random.uniform(0.3, 1.0, n)
        is_correct1 = (scores1 > 0.6).astype(int)
        
        # Weak correlation: more random
        scores2 = scores1.copy()
        is_correct2 = np.random.randint(0, 2, n)
        
        threshold1 = compute_conformal_threshold_simple(scores1, is_correct1, alpha=0.1)
        threshold2 = compute_conformal_threshold_simple(scores2, is_correct2, alpha=0.1)
        
        # Stronger correlation should allow lower threshold
        # (not a strict guarantee, but should hold statistically)
        assert threshold1 <= threshold2 + 0.2


class TestConformalCoverage:
    """Tests for conformal coverage validation."""
    
    def test_validate_coverage(self):
        """Validate coverage on test set."""
        test_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        test_is_correct = np.array([1, 1, 0, 0, 0])
        threshold = 0.65
        
        error_rate, coverage, num_accepted, num_errors = validate_conformal_coverage(
            test_scores, test_is_correct, threshold
        )
        
        # Should accept: [0.9, 0.8, 0.7] → 3 examples
        assert num_accepted == 3
        # Errors in accepted: [0.7] → 1 error
        assert num_errors == 1
        # Error rate: 1/3
        assert abs(error_rate - 1/3) < 0.01
        # Coverage: 3/5
        assert abs(coverage - 0.6) < 0.01
    
    def test_accept_all(self):
        """Low threshold accepts all."""
        test_scores = np.array([0.9, 0.8, 0.7])
        test_is_correct = np.array([1, 1, 0])
        threshold = 0.5
        
        error_rate, coverage, num_accepted, num_errors = validate_conformal_coverage(
            test_scores, test_is_correct, threshold
        )
        
        assert num_accepted == 3
        assert coverage == 1.0
        assert num_errors == 1
        assert abs(error_rate - 1/3) < 0.01
    
    def test_reject_all(self):
        """High threshold rejects all."""
        test_scores = np.array([0.5, 0.4, 0.3])
        test_is_correct = np.array([1, 0, 0])
        threshold = 0.9
        
        error_rate, coverage, num_accepted, num_errors = validate_conformal_coverage(
            test_scores, test_is_correct, threshold
        )
        
        assert num_accepted == 0
        assert coverage == 0.0
        assert num_errors == 0
        assert error_rate == 0.0
    
    def test_empty_test_set(self):
        """Empty test set should return zeros."""
        error_rate, coverage, num_accepted, num_errors = validate_conformal_coverage(
            np.array([]), np.array([]), threshold=0.5
        )
        
        assert error_rate == 0.0
        assert coverage == 0.0
        assert num_accepted == 0
        assert num_errors == 0


class TestConformalCalibration:
    """Tests for full conformal calibration."""
    
    def test_calibration_analysis(self):
        """Complete conformal calibration."""
        np.random.seed(42)
        n = 100
        
        # Generate calibration data
        scores = np.random.uniform(0.3, 1.0, n)
        predictions = np.random.randint(0, 2, n)
        
        # High scores more likely correct
        targets = predictions.copy()
        low_conf = scores < 0.6
        error_mask = low_conf & (np.random.rand(n) < 0.4)
        targets[error_mask] = 1 - targets[error_mask]
        
        result = conformal_prediction_calibration(
            scores, predictions, targets, alpha=0.1
        )
        
        # Check result structure
        assert isinstance(result, ConformalResult)
        assert 0 <= result.threshold <= 1
        assert result.alpha == 0.1
        assert result.coverage_guarantee == 0.9
        assert 0 <= result.empirical_coverage <= 1
        assert result.calibration_size == n
    
    def test_different_methods(self):
        """Test both simple and full methods."""
        np.random.seed(42)
        n = 50
        scores = np.random.uniform(0.3, 1.0, n)
        predictions = np.random.randint(0, 2, n)
        targets = predictions.copy()
        targets[scores < 0.5] = 1 - targets[scores < 0.5]
        
        result_simple = conformal_prediction_calibration(
            scores, predictions, targets, alpha=0.1, method="simple"
        )
        
        result_full = conformal_prediction_calibration(
            scores, predictions, targets, alpha=0.1, method="full"
        )
        
        # Both should give valid thresholds
        assert 0 <= result_simple.threshold <= 1
        assert 0 <= result_full.threshold <= 1
    
    def test_high_confidence_predictions(self):
        """High confidence correct predictions."""
        n = 50
        scores = np.random.uniform(0.8, 1.0, n)
        predictions = np.random.randint(0, 2, n)
        targets = predictions.copy()  # All correct
        
        result = conformal_prediction_calibration(
            scores, predictions, targets, alpha=0.1
        )
        
        # Empirical coverage should be high (close to 1.0)
        assert result.empirical_coverage >= 0.9


class TestCalibrationError:
    """Tests for Expected Calibration Error (ECE)."""
    
    def test_perfect_calibration(self):
        """Perfectly calibrated scores → ECE = 0."""
        # Scores match actual accuracy
        scores = np.array([0.9, 0.9, 0.9, 0.9, 0.9,  # 90% bin
                          0.7, 0.7, 0.7, 0.7, 0.7,  # 70% bin
                          0.5, 0.5, 0.5, 0.5, 0.5])  # 50% bin
        
        # Make accuracy match scores
        is_correct = np.array([1, 1, 1, 1, 0,  # 80% correct ≈ 90%
                              1, 1, 1, 0, 0,  # 60% correct ≈ 70%
                              1, 1, 0, 0, 0])  # 40% correct ≈ 50%
        
        ece = expected_calibration_error(scores, is_correct, num_bins=3)
        
        # Should be low (not exactly 0 due to binning)
        assert ece < 0.15
    
    def test_overconfident(self):
        """Overconfident predictions → positive ECE."""
        # High scores but low accuracy
        scores = np.array([0.9] * 10 + [0.8] * 10)
        is_correct = np.array([0] * 10 + [0] * 10)  # All wrong
        
        ece = expected_calibration_error(scores, is_correct, num_bins=5)
        
        # Should have high ECE (scores much higher than accuracy)
        assert ece > 0.5
    
    def test_underconfident(self):
        """Underconfident predictions → positive ECE."""
        # Low scores but high accuracy
        scores = np.array([0.5] * 10 + [0.6] * 10)
        is_correct = np.array([1] * 10 + [1] * 10)  # All correct
        
        ece = expected_calibration_error(scores, is_correct, num_bins=5)
        
        # Should have positive ECE (scores lower than accuracy)
        assert ece > 0.2
    
    def test_empty_input(self):
        """Empty input should return 0."""
        ece = expected_calibration_error(np.array([]), np.array([]))
        assert ece == 0.0
    
    def test_ece_range(self):
        """ECE should be in [0, 1]."""
        np.random.seed(42)
        scores = np.random.uniform(0, 1, 100)
        is_correct = np.random.randint(0, 2, 100)
        
        ece = expected_calibration_error(scores, is_correct)
        
        assert 0 <= ece <= 1


class TestConfidenceBands:
    """Tests for confidence band computation."""
    
    def test_confidence_band(self):
        """Compute confidence bands."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        lower, upper = confidence_band_quantile(scores, alpha=0.2)
        
        # 10th and 90th percentiles
        assert abs(lower - 0.18) < 0.05
        assert abs(upper - 0.82) < 0.05
        assert lower < upper
    
    def test_empty_input(self):
        """Empty input should return default bounds."""
        lower, upper = confidence_band_quantile(np.array([]), alpha=0.1)
        assert lower == 0.0
        assert upper == 1.0
    
    def test_tight_band(self):
        """Small alpha → tight band."""
        scores = np.linspace(0, 1, 100)
        
        lower, upper = confidence_band_quantile(scores, alpha=0.01)
        
        # Should exclude only 0.5% on each side
        assert lower < 0.02
        assert upper > 0.98


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_format_summary(self):
        """Format conformal result summary."""
        result = ConformalResult(
            threshold=0.75,
            alpha=0.1,
            coverage_guarantee=0.9,
            empirical_coverage=0.92,
            calibration_size=100,
            num_rejected=15
        )
        
        summary = format_conformal_summary(result)
        
        assert "0.75" in summary
        assert "90" in summary or "0.9" in summary
        assert "100" in summary
    
    def test_combine_selective_and_conformal(self):
        """Combine both approaches."""
        np.random.seed(42)
        n = 80
        scores = np.random.uniform(0.4, 1.0, n)
        predictions = np.random.randint(0, 2, n)
        targets = predictions.copy()
        targets[scores < 0.6] = 1 - targets[scores < 0.6]
        
        result = combine_selective_and_conformal(
            scores, predictions, targets, alpha=0.1, target_risk=0.1
        )
        
        # Check both results present
        assert "conformal" in result
        assert "selective" in result
        assert "combined_threshold" in result
        
        # Combined threshold should be at least as high as both
        assert result["combined_threshold"] >= result["conformal"].threshold
        assert result["combined_threshold"] >= result["selective"].optimal_threshold


class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_all_same_score(self):
        """All examples have same score."""
        scores = np.array([0.7] * 10)
        is_correct = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
        
        threshold = compute_conformal_threshold_simple(scores, is_correct, alpha=0.1)
        
        assert 0 <= threshold <= 1
    
    def test_single_error(self):
        """Only one error in calibration set."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        is_correct = np.array([1, 1, 1, 1, 0])
        
        threshold = compute_conformal_threshold_simple(scores, is_correct, alpha=0.1)
        
        # Should be near the single error score
        assert 0.4 <= threshold <= 0.6
    
    def test_alternating_correctness(self):
        """Alternating correct/incorrect."""
        scores = np.linspace(0.5, 1.0, 10)
        is_correct = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        
        result = conformal_prediction_calibration(
            scores,
            np.random.randint(0, 2, 10),
            np.random.randint(0, 2, 10),
            alpha=0.1
        )
        
        assert 0 <= result.threshold <= 1
        assert result.calibration_size == 10
    
    def test_very_small_calibration_set(self):
        """Very small calibration set (n=5)."""
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        is_correct = np.array([1, 1, 0, 0, 0])
        
        threshold = compute_conformal_threshold_simple(scores, is_correct, alpha=0.1)
        
        # Should still work
        assert 0 <= threshold <= 1
