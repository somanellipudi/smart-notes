"""
Paired significance tests for comparing prediction methods.

Implements:
1. McNemar test (exact or chi-square with continuity correction)
2. Permutation test (paired accuracy difference)

Both tests are deterministic and suitable for small to medium datasets.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, Any]:
    """
    Perform McNemar test comparing two paired classifiers.
    
    McNemar's test assesses whether two paired classifiers differ significantly.
    It focuses on cases where the classifiers disagree.
    
    Args:
        y_true: Ground truth labels (1D array)
        pred_a: Predictions from method A (1D array)
        pred_b: Predictions from method B (1D array)
    
    Returns:
        Dictionary with:
        - test: "mcnemar"
        - n01: Count of (A wrong, B correct) pairs
        - n10: Count of (A correct, B wrong) pairs
        - chi2: Chi-squared test statistic
        - p_value: Two-tailed p-value
        - significant_0.05: Boolean for significance at α=0.05
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    
    # Compute correctness for each method
    correct_a = (pred_a == y_true).astype(int)
    correct_b = (pred_b == y_true).astype(int)
    
    # Build contingency table: disagreement cells only
    # n01: A is wrong, B is correct
    n01 = np.sum((correct_a == 0) & (correct_b == 1))
    # n10: A is correct, B is wrong
    n10 = np.sum((correct_a == 1) & (correct_b == 0))
    # n00: Both wrong (not used in McNemar)
    # n11: Both correct (not used in McNemar)
    
    n_disagreement = n01 + n10
    
    # Special case: no disagreement → p-value = 1.0
    if n_disagreement == 0:
        return {
            "test": "mcnemar",
            "n01": int(n01),
            "n10": int(n10),
            "chi2": None,
            "p_value": 1.0,
            "significant_0.05": False
        }
    
    # Choose exact or approximate test
    if n_disagreement <= 25:
        # Exact McNemar test using binomial distribution
        # Under null: P(n01 | n_disagreement) ~ Binomial(n_disagreement, 0.5)
        p_value = 2 * stats.binom.sf(max(n01, n10) - 1, n_disagreement, 0.5)
        # Clamp to [0, 1] (binom.sf can sometimes exceed 1 due to numerical precision)
        p_value = min(p_value, 1.0)
        chi2_stat = np.nan  # Not applicable for exact test
    else:
        # Chi-square with continuity correction
        # chi2 = ((|n01 - n10| - 1)^2) / (n01 + n10)
        chi2_stat = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    
    return {
        "test": "mcnemar",
        "n01": int(n01),
        "n10": int(n10),
        "chi2": float(chi2_stat) if not np.isnan(chi2_stat) else None,
        "p_value": float(p_value),
        "significant_0.05": float(p_value) < 0.05
    }


def permutation_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_iter: int = 10000,
    seed: int = 0
) -> Dict[str, Any]:
    """
    Perform permutation test on paired accuracy difference.
    
    Permutation test generates null distribution by randomly swapping predictions
    per sample and computing test statistic. P-value is proportion of permutations
    with test statistic >= observed.
    
    Args:
        y_true: Ground truth labels (1D array)
        pred_a: Predictions from method A (1D array)
        pred_b: Predictions from method B (1D array)
        n_iter: Number of permutation iterations
        seed: Random seed for determinism
    
    Returns:
        Dictionary with:
        - test: "permutation"
        - accuracy_diff: Observed difference in accuracy (A - B)
        - p_value: One-tailed p-value (proportion >= observed diff)
        - n_iter: Number of permutations performed
        - significant_0.05: Boolean for significance at α=0.05
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    
    rng = np.random.RandomState(seed)
    
    # Compute observed accuracy difference
    acc_a = np.mean(pred_a == y_true)
    acc_b = np.mean(pred_b == y_true)
    observed_diff = acc_a - acc_b
    
    # Generate null distribution
    n_samples = len(y_true)
    permuted_diffs = np.zeros(n_iter)
    
    for i in range(n_iter):
        # Randomly swap predictions per sample (under permutation)
        swap_mask = rng.rand(n_samples) < 0.5
        perm_a = pred_a.copy()
        perm_b = pred_b.copy()
        
        # Swap predictions for selected samples
        perm_a[swap_mask], perm_b[swap_mask] = perm_b[swap_mask].copy(), perm_a[swap_mask].copy()
        
        # Compute accuracy difference
        acc_a_perm = np.mean(perm_a == y_true)
        acc_b_perm = np.mean(perm_b == y_true)
        permuted_diffs[i] = acc_a_perm - acc_b_perm
    
    # Compute one-tailed p-value
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    
    return {
        "test": "permutation",
        "accuracy_diff": float(observed_diff),
        "p_value": float(p_value),
        "n_iter": int(n_iter),
        "significant_0.05": float(p_value) < 0.05
    }


def run_paired_tests(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_perm_iter: int = 10000,
    seed: int = 0
) -> Dict[str, Any]:
    """
    Run both McNemar and permutation tests.
    
    Args:
        y_true: Ground truth labels
        pred_a: Predictions from method A
        pred_b: Predictions from method B
        n_perm_iter: Number of permutation iterations
        seed: Random seed
    
    Returns:
        Dictionary with results from both tests plus accuracy difference.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    
    acc_a = np.mean(pred_a == y_true)
    acc_b = np.mean(pred_b == y_true)
    
    return {
        "accuracy_a": float(acc_a),
        "accuracy_b": float(acc_b),
        "accuracy_diff": float(acc_a - acc_b),
        "mcnemar": mcnemar_test(y_true, pred_a, pred_b),
        "permutation": permutation_test(y_true, pred_a, pred_b, n_iter=n_perm_iter, seed=seed)
    }
