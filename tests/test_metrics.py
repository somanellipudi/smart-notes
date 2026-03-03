"""
Unit tests for unified metrics module.
"""

import numpy as np
import pytest
from src.eval.metrics import MetricsComputer


class TestECEComputation:
    """Tests for ECE (Expected Calibration Error) computation."""
    
    def test_ece_perfect_calibration(self):
        """ECE should be 0 for perfectly calibrated predictions."""
        # Perfect calibration: confidence = accuracy in each bin
        # Note: confidence = max(p, 1-p), so we need to create data properly
        np.random.seed(42)
        n = 1000
        
        # Create perfectly calibrated data
        bins = 10
        samples_per_bin = n // bins
        
        probs = []
        labels = []
        
        for b in range(bins):
            # For bin b, use confidence threshold at (b+0.5)/bins
            # Confidence = max(p, 1-p), so for high confidence we need p close to 1 or 0
            conf_target = (b + 0.5) / bins  # Target confidence
            
            if conf_target > 0.5:
                # Use probabilities close to 1 (predicted class 1)
                prob_val = conf_target
                n_correct = int(samples_per_bin * conf_target)
                
                # Add correct predictions (label=1, prob≈conf_target)
                probs.extend([prob_val] * n_correct)
                labels.extend([1] * n_correct)
                
                # Add incorrect predictions (label=0, prob≈conf_target)
                n_wrong = samples_per_bin - n_correct
                probs.extend([prob_val] * n_wrong)
                labels.extend([0] * n_wrong)
            else:
                # Use low probabilities
                prob_val = 1.0 - conf_target
                n_correct = int(samples_per_bin * conf_target)
                
                probs.extend([prob_val] * n_correct)
                labels.extend([0] * n_correct)
                
                n_wrong = samples_per_bin - n_correct
                probs.extend([prob_val] * n_wrong)
                labels.extend([1] * n_wrong)
        
        probs = np.array(probs)
        labels = np.array(labels)
        
        computer = MetricsComputer(n_bins=bins)
        result = computer.compute_ece(probs, labels)
        
        # Should be close to 0 (perfect calibration)
        assert result['ece'] < 0.10, f"ECE should be low for calibrated data, got {result['ece']}"
    
    def test_ece_overconfident(self):
        """ECE increases when predictions are overconfident."""
        np.random.seed(42)
        n = 100
        
        # Overconfident: always predict high confidence, but many wrong
        probs = np.random.uniform(0.7, 1.0, n)
        labels = np.random.binomial(1, 0.5, n)  # 50% correct
        
        computer = MetricsComputer(n_bins=10)
        result = computer.compute_ece(probs, labels)
        
        # Should have high ECE (confidence >> accuracy)
        assert result['ece'] > 0.15
    
    def test_ece_bounds(self):
        """ECE should be in [0, 1]."""
        np.random.seed(42)
        
        for _ in range(10):
            n = np.random.randint(50, 200)
            probs = np.random.uniform(0, 1, n)
            labels = np.random.binomial(1, 0.5, n)
            
            computer = MetricsComputer(n_bins=10)
            result = computer.compute_ece(probs, labels)
            
            assert 0 <= result['ece'] <= 1.0, f"ECE {result['ece']} out of bounds"
    
    def test_ece_bin_statistics(self):
        """ECE bin statistics should be correctly computed."""
        np.random.seed(42)
        n = 1000
        
        probs = np.random.uniform(0, 1, n)
        labels = (probs > 0.5).astype(int)  # Perfectly calibrated
        
        computer = MetricsComputer(n_bins=10)
        result = computer.compute_ece(probs, labels, return_bins=True)
        
        assert 'bins' in result
        assert len(result['bins']) > 0
        
        # Sum of bin weights should be 1.0
        total_weight = sum(b['count'] for b in result['bins']) / n
        assert abs(total_weight - 1.0) < 0.01


class TestAccuracyCoverageCurve:
    """Tests for accuracy-coverage curve computation."""
    
    def test_coverage_range(self):
        """Coverage should be in [0, 1] and decrease with threshold."""
        np.random.seed(42)
        n = 200
        
        confidences = np.random.uniform(0, 1, n)
        correctness = np.random.binomial(1, 0.7, n)
        
        computer = MetricsComputer()
        result = computer.compute_accuracy_coverage_curve(
            confidences, correctness,
            thresholds=np.linspace(0.5, 1.0, 11)
        )
        
        # All coverage values should be in [0, 1]
        coverage = np.array(result['coverage'])
        assert np.all(coverage >= 0) and np.all(coverage <= 1.0)
        
        # Coverage should decrease with threshold
        assert np.all(np.diff(coverage) <= 0), "Coverage should decrease or stay same with threshold"
    
    def test_accuracy_range(self):
        """Selective accuracy should be in [0, 1]."""
        np.random.seed(42)
        n = 200
        
        confidences = np.random.uniform(0, 1, n)
        correctness = np.random.binomial(1, 0.8, n)
        
        computer = MetricsComputer()
        result = computer.compute_accuracy_coverage_curve(confidences, correctness)
        
        accuracy = np.array(result['accuracy'])
        assert np.all(accuracy >= 0) and np.all(accuracy <= 1.0)
    
    def test_accuracy_increases_with_threshold(self):
        """Selective accuracy should generally increase with threshold."""
        np.random.seed(42)
        n = 300
        
        # Create data where higher confidence correlates with correctness
        confidences = np.random.uniform(0, 1, n)
        correctness = (confidences > 0.5).astype(int)
        
        computer = MetricsComputer()
        result = computer.compute_accuracy_coverage_curve(
            confidences, correctness,
            thresholds=np.linspace(0.5, 1.0, 11)
        )
        
        accuracy = np.array(result['accuracy'])
        # Most of the increases should be non-negative
        diffs = np.diff(accuracy)
        non_negative = np.sum(diffs >= 0)
        assert non_negative >= len(diffs) - 2, "Most accuracy increases should be non-negative"


class TestAUCAC:
    """Tests for AUC-AC (Area Under Accuracy-Coverage) computation."""
    
    def test_auc_ac_bounds(self):
        """AUC-AC should be in [0, 1]."""
        np.random.seed(42)
        
        for _ in range(10):
            n_points = np.random.randint(5, 50)
            coverage = np.sort(np.random.uniform(0, 1, n_points))
            accuracy = np.random.uniform(0, 1, n_points)
            
            computer = MetricsComputer()
            auc_ac = computer.compute_auc_ac(coverage, accuracy)
            
            assert 0 <= auc_ac <= 1.0, f"AUC-AC {auc_ac} out of bounds"
    
    def test_auc_ac_perfect(self):
        """AUC-AC should be high for good predictions."""
        np.random.seed(42)
        
        # Perfect selective prediction: high selectivity
        coverage = np.linspace(1.0, 0.0, 11)  # Decreasing from 1 to 0
        accuracy = np.ones(11)  # Always correct
        
        computer = MetricsComputer()
        auc_ac = computer.compute_auc_ac(coverage, accuracy)
        
        assert auc_ac > 0.8, f"AUC-AC should be high for perfect predictions, got {auc_ac}"
    
    def test_auc_ac_random(self):
        """AUC-AC should be low for random predictions."""
        np.random.seed(42)
        
        # Random: accuracy constant as coverage varies
        coverage = np.linspace(1.0, 0.0, 11)
        accuracy = np.full(11, 0.5)  # 50% (random)
        
        computer = MetricsComputer()
        auc_ac = computer.compute_auc_ac(coverage, accuracy)
        
        assert auc_ac < 0.6, f"AUC-AC should be low for random predictions, got {auc_ac}"


class TestAllMetrics:
    """Tests for compute_all_metrics convenience function."""
    
    def test_all_metrics_shape(self):
        """compute_all_metrics should return all required metrics."""
        np.random.seed(42)
        n = 200
        
        probs = np.random.uniform(0, 1, n)
        labels = np.random.binomial(1, 0.6, n)
        
        computer = MetricsComputer()
        result = computer.compute_all_metrics(probs, labels)
        
        # Check required keys
        assert 'accuracy' in result
        assert 'ece' in result
        assert 'auc_ac' in result
        assert 'macro_f1' in result
        assert 'ece_bins' in result
        assert 'accuracy_coverage_curve' in result
        assert 'metadata' in result
    
    def test_all_metrics_values_reasonable(self):
        """All metrics should have reasonable values."""
        np.random.seed(42)
        n = 300
        
        probs = np.random.uniform(0.3, 0.7, n)  # Mediocre confidence
        labels = (probs > 0.5).astype(int)  # Somewhat correlated
        
        computer = MetricsComputer()
        result = computer.compute_all_metrics(probs, labels)
        
        # Check ranges
        assert 0 <= result['accuracy'] <= 1.0
        assert 0 <= result['ece'] <= 1.0
        assert 0 <= result['auc_ac'] <= 1.0
        assert 0 <= result['macro_f1'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
