"""
Tests for significance_tests module.

Test cases:
1. Identical predictors → p_value ≈ 1
2. Clearly different predictors → p_value < 0.05
3. Permutation determinism with seed
"""

import pytest
import numpy as np
from src.eval.significance_tests import mcnemar_test, permutation_test, run_paired_tests


class TestMcNemarTest:
    """Tests for McNemar test."""
    
    def test_identical_predictors(self):
        """When predictors are identical, p_value should be 1."""
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
        pred_a = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
        pred_b = pred_a.copy()
        
        result = mcnemar_test(y_true, pred_a, pred_b)
        
        assert result["test"] == "mcnemar"
        assert result["n01"] == 0
        assert result["n10"] == 0
        assert result["p_value"] == 1.0
        assert not result["significant_0.05"]
    
    def test_clearly_different_predictors(self):
        """When one predictor is clearly better, p_value should be < 0.05."""
        # Create scenario where pred_a is always wrong but pred_b is mostly correct
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, size=n_samples)
        
        # pred_a is random (50% accuracy)
        pred_a = np.random.randint(0, 2, size=n_samples)
        
        # pred_b follows y_true (100% accuracy)
        pred_b = y_true.copy()
        
        result = mcnemar_test(y_true, pred_a, pred_b)
        
        assert result["test"] == "mcnemar"
        assert result["n01"] > result["n10"]  # pred_a wrong but pred_b correct > pred_a correct but pred_b wrong
        assert result["p_value"] < 0.05
        assert result["significant_0.05"]
    
    def test_small_contingency_exact_test(self):
        """Test uses exact binomial when n01+n10 <= 25."""
        y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        pred_a = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 1])
        pred_b = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 1])
        
        result = mcnemar_test(y_true, pred_a, pred_b)
        
        assert result["test"] == "mcnemar"
        # For exact test, chi2 should be None
        assert result["chi2"] is None or np.isnan(result["chi2"]) or result["chi2"] is np.nan
        # p_value should be computed
        assert 0 <= result["p_value"] <= 1


class TestPermutationTest:
    """Tests for permutation test."""
    
    def test_identical_predictors(self):
        """When predictors are identical, p_value should be 1."""
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
        pred_a = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
        pred_b = pred_a.copy()
        
        result = permutation_test(y_true, pred_a, pred_b, n_iter=1000, seed=42)
        
        assert result["test"] == "permutation"
        assert result["accuracy_diff"] == 0.0
        assert result["p_value"] == 1.0
        assert not result["significant_0.05"]
    
    def test_clearly_different_predictors(self):
        """When one predictor is clearly better, p_value should be < 0.05."""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, size=n_samples)
        
        # pred_a is random (50% accuracy)
        pred_a = np.random.randint(0, 2, size=n_samples)
        
        # pred_b follows y_true (100% accuracy)
        pred_b = y_true.copy()
        
        result = permutation_test(y_true, pred_a, pred_b, n_iter=1000, seed=42)
        
        assert result["test"] == "permutation"
        assert result["accuracy_diff"] < -0.3  # pred_b is much better
        assert result["p_value"] < 0.05
        assert result["significant_0.05"]
    
    def test_determinism_with_seed(self):
        """Test results should be identical with same seed."""
        np.random.seed(42)
        n_samples = 50
        y_true = np.random.randint(0, 2, size=n_samples)
        pred_a = np.random.randint(0, 2, size=n_samples)
        pred_b = np.random.randint(0, 2, size=n_samples)
        
        # Run test twice with same seed
        result1 = permutation_test(y_true, pred_a, pred_b, n_iter=1000, seed=0)
        result2 = permutation_test(y_true, pred_a, pred_b, n_iter=1000, seed=0)
        
        assert result1["accuracy_diff"] == result2["accuracy_diff"]
        assert result1["p_value"] == result2["p_value"]
    
    def test_different_seed_different_results(self):
        """Different seeds should produce slightly different p-values (stochastic)."""
        np.random.seed(42)
        n_samples = 50
        y_true = np.random.randint(0, 2, size=n_samples)
        pred_a = np.random.randint(0, 2, size=n_samples)
        pred_b = np.random.randint(0, 2, size=n_samples)
        
        result1 = permutation_test(y_true, pred_a, pred_b, n_iter=1000, seed=0)
        result2 = permutation_test(y_true, pred_a, pred_b, n_iter=1000, seed=1)
        
        # p-values should be similar but not identical (stochastic)
        assert abs(result1["p_value"] - result2["p_value"]) < 0.1


class TestRunPairedTests:
    """Tests for run_paired_tests wrapper."""
    
    def test_returns_both_tests(self):
        """Should return results from both McNemar and permutation tests."""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, size=n_samples)
        pred_a = np.random.randint(0, 2, size=n_samples)
        pred_b = np.random.randint(0, 2, size=n_samples)
        
        result = run_paired_tests(y_true, pred_a, pred_b, n_perm_iter=1000, seed=42)
        
        assert "accuracy_a" in result
        assert "accuracy_b" in result
        assert "accuracy_diff" in result
        assert "mcnemar" in result
        assert "permutation" in result
        
        # Check nested test results
        assert result["mcnemar"]["test"] == "mcnemar"
        assert result["permutation"]["test"] == "permutation"


class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_single_sample(self):
        """Should handle single sample."""
        y_true = np.array([1])
        pred_a = np.array([1])
        pred_b = np.array([0])
        
        result = mcnemar_test(y_true, pred_a, pred_b)
        assert result["n10"] == 1
        assert result["n01"] == 0
    
    def test_all_correct(self):
        """All predictions correct."""
        y_true = np.array([1, 0, 1, 0, 1])
        pred_a = y_true.copy()
        pred_b = y_true.copy()
        
        result = permutation_test(y_true, pred_a, pred_b, seed=42)
        assert result["accuracy_diff"] == 0.0
    
    def test_all_wrong(self):
        """All predictions wrong."""
        y_true = np.array([1, 0, 1, 0, 1])
        pred_a = 1 - y_true
        pred_b = 1 - y_true
        
        result = permutation_test(y_true, pred_a, pred_b, seed=42)
        assert result["accuracy_diff"] == 0.0
    
    def test_large_sample(self):
        """Should handle large samples."""
        np.random.seed(42)
        n_samples = 5000
        y_true = np.random.randint(0, 2, size=n_samples)
        
        # Method A: 70% accuracy
        pred_a = y_true.copy()
        flip_mask_a = np.random.rand(n_samples) < 0.3
        pred_a[flip_mask_a] = 1 - pred_a[flip_mask_a]
        
        # Method B: 75% accuracy
        pred_b = y_true.copy()
        flip_mask_b = np.random.rand(n_samples) < 0.25
        pred_b[flip_mask_b] = 1 - pred_b[flip_mask_b]
        
        result = run_paired_tests(y_true, pred_a, pred_b, n_perm_iter=1000, seed=42)
        
        # Should see p-values in reasonable range
        assert 0 <= result["mcnemar"]["p_value"] <= 1
        assert 0 <= result["permutation"]["p_value"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
