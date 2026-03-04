#!/usr/bin/env python3
"""
Statistical significance testing and class imbalance analysis.

Computes:
- Paired bootstrap test for baseline comparisons
- McNemar's test for pairwise differences
- Balanced accuracy (macro average)
- Per-class calibration metrics
- Class distribution analysis

Tests whether observed differences between systems are statistically significant
given the small test set (n=260).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

try:
    from scipy.stats import binom_test
except ImportError:
    # For scipy >= 1.7
    from scipy.stats import binomtest
    def binom_test(x, n, p, alternative='two-sided'):
        result = binomtest(x, n, p, alternative=alternative)
        return result.pvalue


def compute_balanced_accuracy(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """
    Balanced accuracy: macro average of per-class recalls.
    
    BA = (recall_class_0 + recall_class_1) / 2
    
    More robust than accuracy when classes are imbalanced.
    """
    unique_classes = np.unique(y_true)
    recalls = []
    
    for class_label in unique_classes:
        mask = y_true == class_label
        if np.sum(mask) == 0:
            continue
        
        recall = np.mean(predictions[mask] == y_true[mask])
        recalls.append(recall)
    
    return float(np.mean(recalls)) if recalls else 0.0


def compute_per_class_metrics(y_true: np.ndarray, predictions: np.ndarray) -> Dict:
    """Compute precision, recall, F1 per class."""
    
    unique_classes = np.unique(y_true)
    metrics = {}
    
    for class_label in unique_classes:
        y_mask = y_true == class_label
        pred_mask = predictions == class_label
        
        tp = np.sum(y_mask & pred_mask)
        fp = np.sum(~y_mask & pred_mask)
        fn = np.sum(y_mask & ~pred_mask)
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        metrics[f"class_{class_label}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(y_mask))
        }
    
    return metrics


def compute_per_class_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Dict:
    """Compute ECE separately for each class."""
    
    unique_classes = np.unique(y_true)
    ece_results = {}
    
    for class_label in unique_classes:
        mask = y_true == class_label
        if np.sum(mask) < 10:  # Skip if too few samples
            continue
        
        y_class = y_true[mask]
        probs_class = probs[mask]
        
        # For this class: confidence = P(class_label)
        # For binary: probs is P(class_1), so for class_0 use 1-probs
        if class_label == 0:
            confidence_class = 1 - probs_class
        else:
            confidence_class = probs_class
        
        # Compute ECE
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_count = 0
        
        for i in range(n_bins):
            mask_bin = (confidence_class >= bin_edges[i]) & (confidence_class < bin_edges[i+1])
            if i == n_bins - 1:
                mask_bin = (confidence_class >= bin_edges[i]) & (confidence_class <= bin_edges[i+1])
            
            if not np.any(mask_bin):
                continue
            
            bin_weight = np.sum(mask_bin) / len(y_class)
            y_bin = y_class[mask_bin]
            conf_bin = confidence_class[mask_bin]
            
            # For this class, empirical accuracy = fraction correctly predicted
            empirical_acc = np.mean(y_bin == class_label)
            avg_conf = np.mean(conf_bin)
            
            ece += bin_weight * abs(avg_conf - empirical_acc)
            bin_count += 1
        
        ece_results[f"class_{class_label}"] = {
            "ece": float(ece),
            "n_samples": int(np.sum(mask)),
            "n_bins": bin_count
        }
    
    return ece_results


def mcnemar_test(y_true: np.ndarray, predictions_a: np.ndarray, predictions_b: np.ndarray) -> Dict:
    """
    McNemar's test for comparing two classifiers on the same test set.
    
    Tests whether there's a significant difference in error rates between two systems.
    """
    
    errors_a = y_true != predictions_a
    errors_b = y_true != predictions_b
    
    # McNemar's contingency table
    disagreement_on_a = errors_a & ~errors_b  # A is wrong, B is right
    disagreement_on_b = ~errors_a & errors_b  # A is right, B is wrong
    
    n01 = np.sum(disagreement_on_a)  # A wrong, B right
    n10 = np.sum(disagreement_on_b)  # A right, B wrong
    
    # McNemar's statistic: chi2 = (n01 - n10)^2 / (n01 + n10)
    total = n01 + n10
    
    if total == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "interpretation": "No disagreement between systems; cannot compute McNemar test"
        }
    
    # Exact binomial test is more appropriate for small samples
    # H0: P(disagreement_on_a) = P(disagreement_on_b) = 0.5
    p_value = binom_test(n01, total, 0.5, alternative='two-sided')
    
    chi2_stat = (n01 - n10) ** 2 / (total + 1e-10)
    
    return {
        "n_disagreements": total,
        "a_wrong_b_right": int(n01),
        "a_right_b_wrong": int(n10),
        "chi2_statistic": float(chi2_stat),
        "p_value_binomial": float(p_value),
        "significant_at_0.05": p_value < 0.05,
        "interpretation": ("Significant difference detected" if p_value < 0.05 
                          else "No significant difference detected")
    }


def paired_bootstrap_comparison(
    y_true_a: np.ndarray,
    probs_a: np.ndarray,
    y_true_b: np.ndarray,
    probs_b: np.ndarray,
    n_bootstraps: int = 2000
) -> Dict:
    """
    Paired bootstrap test for two systems evaluated on same test set.
    
    Estimates confidence intervals for accuracy difference.
    """
    
    if len(y_true_a) != len(y_true_b):
        raise ValueError("Predictions must have same length")
    
    preds_a = (probs_a >= 0.5).astype(int)
    preds_b = (probs_b >= 0.5).astype(int)
    
    acc_a = np.mean(y_true_a == preds_a)
    acc_b = np.mean(y_true_b == preds_b)
    acc_diff = acc_a - acc_b
    
    # Bootstrap to get CI on difference
    n_samples = len(y_true_a)
    diffs = []
    
    np.random.seed(42)
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        y_a_boot = y_true_a[indices]
        y_b_boot = y_true_b[indices]
        preds_a_boot = preds_a[indices]
        preds_b_boot = preds_b[indices]
        
        acc_a_boot = np.mean(y_a_boot == preds_a_boot)
        acc_b_boot = np.mean(y_b_boot == preds_b_boot)
        
        diffs.append(acc_a_boot - acc_b_boot)
    
    diffs = np.array(diffs)
    
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    ci_includes_zero = ci_lower <= 0 <= ci_upper
    
    return {
        "accuracy_a": float(acc_a),
        "accuracy_b": float(acc_b),
        "accuracy_difference": float(acc_diff),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "ci_includes_zero": ci_includes_zero,
        "significantly_different": not ci_includes_zero,
        "interpretation": ("No significant difference" if ci_includes_zero 
                          else "Significant difference (CI excludes zero)")
    }


def analyze_class_imbalance(y_true: np.ndarray) -> Dict:
    """Analyze class distribution and imbalance metrics."""
    
    unique_classes, counts = np.unique(y_true, return_counts=True)
    
    # Class distribution
    class_dist = {
        f"class_{cls}": {
            "count": int(cnt),
            "percentage": float(cnt / len(y_true) * 100)
        }
        for cls, cnt in zip(unique_classes, counts)
    }
    
    # Imbalance ratio
    imbalance_ratio = max(counts) / (min(counts) + 1e-10)
    
    # Proportions
    proportions = counts / len(y_true)
    
    return {
        "n_classes": int(len(unique_classes)),
        "n_total_samples": int(len(y_true)),
        "class_distribution": class_dist,
        "imbalance_ratio": float(imbalance_ratio),
        "imbalance_interpretation": ("Balanced" if imbalance_ratio < 1.2
                                    else "Moderately imbalanced" if imbalance_ratio < 2.0
                                    else "Highly imbalanced")
    }


def main():
    """Main: compute significance tests and class imbalance metrics."""
    
    pred_file = Path("artifacts/preds/CalibraTeach.npz")
    output_dir = Path("artifacts")
    output_file = output_dir / "statistical_significance_analysis.json"
    
    if not pred_file.exists():
        print(f"ERROR: Prediction file not found: {pred_file}")
        return
    
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE & CLASS IMBALANCE ANALYSIS")
    print("="*70 + "\n")
    
    data = np.load(pred_file, allow_pickle=True)
    y_true = data['y_true']
    probs = data['probs']
    predictions = (probs >= 0.5).astype(int)
    
    results = {
        "metadata": {
            "n_samples": int(len(y_true)),
            "pred_file": str(pred_file),
        }
    }
    
    # 1. Class imbalance analysis
    print("CLASS IMBALANCE ANALYSIS")
    print("-" * 70)
    
    imbalance = analyze_class_imbalance(y_true)
    results["class_imbalance"] = imbalance
    
    for cls, dist in imbalance["class_distribution"].items():
        print(f"  {cls}: {dist['count']} samples ({dist['percentage']:.1f}%)")
    
    print(f"  Imbalance ratio: {imbalance['imbalance_ratio']:.2f}x")
    print(f"  Status: {imbalance['imbalance_interpretation']}\n")
    
    # 2. Balanced accuracy
    print("BALANCED ACCURACY")
    print("-" * 70)
    
    ba = compute_balanced_accuracy(y_true, predictions)
    standard_acc = np.mean(y_true == predictions)
    
    print(f"  Standard Accuracy: {standard_acc:.4f}")
    print(f"  Balanced Accuracy: {ba:.4f}")
    print(f"  Difference: {standard_acc - ba:.4f}\n")
    
    results["balanced_accuracy"] = {
        "value": ba,
        "standard_accuracy": standard_acc,
        "description": "Macro-average recall across classes (robust to imbalance)"
    }
    
    # 3. Per-class metrics
    print("PER-CLASS PERFORMANCE")
    print("-" * 70)
    
    per_class = compute_per_class_metrics(y_true, predictions)
    results["per_class_metrics"] = per_class
    
    for cls, metrics in per_class.items():
        print(f"  {cls}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, " +
              f"F1={metrics['f1']:.3f}, support={metrics['support']}")
    
    print()
    
    # 4. Per-class calibration
    print("PER-CLASS ECE")
    print("-" * 70)
    
    per_class_ece = compute_per_class_ece(y_true, probs, n_bins=10)
    results["per_class_ece"] = per_class_ece
    
    for cls, ece_info in per_class_ece.items():
        print(f"  {cls}: ECE={ece_info['ece']:.4f} (n={ece_info['n_samples']})")
    
    print(f"\n✓ Significance and imbalance analysis complete")
    
    # Save results
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_file}\n")
    
    return results


if __name__ == "__main__":
    main()
