#!/usr/bin/env python3
"""
Compute advanced calibration metrics for manuscript revision.

Computes:
- Brier Score
- Negative Log-Likelihood (NLL)
- Adaptive ECE (equal-mass binning)
- ECE with multiple bin counts (10, 15, 20)
- Per-bin calibration details

All metrics computed from existing prediction artifacts.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List


def compute_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Brier score: mean squared error between predicted and empirical probabilities.
    BS = (1/n) * sum((p_i - y_i)^2)
    
    Range: [0, 1], where 0 is perfect calibration.
    """
    return float(np.mean((probs - y_true) ** 2))


def compute_nll(y_true: np.ndarray, probs: np.ndarray, epsilon: float = 1e-15) -> float:
    """
    Negative Log-Likelihood (cross-entropy).
    NLL = -(1/n) * sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))
    
    Lower is better. Commonly used in calibration literature.
    """
    probs = np.clip(probs, epsilon, 1 - epsilon)
    nll = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
    return float(nll)


def compute_ece_adaptive(
    y_true: np.ndarray, 
    probs: np.ndarray, 
    n_bins: int = 10
) -> Tuple[float, List[Dict]]:
    """
    Adaptive ECE using equal-mass (equal-count) binning.
    
    Divides predictions into equally-sized bins by percentile rather than by
    confidence value ranges. This is more robust to skewed confidence distributions.
    
    Returns:
        ECE value and list of bin statistics
    """
    # Confidence: max(p, 1-p)
    confidence = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(int)
    correctness = (predictions == y_true).astype(float)
    
    # Sort by confidence
    sorted_indices = np.argsort(confidence)
    sorted_conf = confidence[sorted_indices]
    sorted_correct = correctness[sorted_indices]
    
    # Create equal-sized bins
    bin_size = len(y_true) // n_bins
    ece_value = 0.0
    bin_stats = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)
        
        bin_conf = sorted_conf[start_idx:end_idx]
        bin_correct = sorted_correct[start_idx:end_idx]
        
        if len(bin_correct) == 0:
            continue
        
        avg_conf = float(np.mean(bin_conf))
        empirical_acc = float(np.mean(bin_correct))
        calibration_gap = abs(avg_conf - empirical_acc)
        bin_weight = len(bin_correct) / len(y_true)
        
        ece_value += bin_weight * calibration_gap
        
        bin_stats.append({
            "bin": i + 1,
            "size": len(bin_correct),
            "avg_confidence": avg_conf,
            "empirical_accuracy": empirical_acc,
            "calibration_gap": calibration_gap,
            "weight": bin_weight
        })
    
    return float(ece_value), bin_stats


def compute_ece_equal_width(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, List[Dict]]:
    """
    Standard ECE with equal-width bins (equal confidence ranges).
    
    This is the standard definition used in most calibration papers.
    
    Returns:
        ECE value and list of bin statistics
    """
    confidence = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(int)
    correctness = (predictions == y_true).astype(float)
    
    # Create equal-width bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_value = 0.0
    bin_stats = []
    
    for i in range(n_bins):
        mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
        if i == n_bins - 1:  # Include upper edge in last bin
            mask = (confidence >= bin_edges[i]) & (confidence <= bin_edges[i + 1])
        
        if not np.any(mask):
            continue
        
        bin_conf = confidence[mask]
        bin_correct = correctness[mask]
        
        avg_conf = float(np.mean(bin_conf))
        empirical_acc = float(np.mean(bin_correct))
        calibration_gap = abs(avg_conf - empirical_acc)
        bin_weight = np.sum(mask) / len(y_true)
        
        ece_value += bin_weight * calibration_gap
        
        bin_stats.append({
            "bin": i + 1,
            "confidence_range": f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]",
            "size": int(np.sum(mask)),
            "avg_confidence": avg_conf,
            "empirical_accuracy": empirical_acc,
            "calibration_gap": calibration_gap,
            "weight": bin_weight
        })
    
    return float(ece_value), bin_stats


def compute_calibration_metrics(pred_file: Path) -> Dict:
    """Load predictions and compute all advanced calibration metrics."""
    
    print(f"Loading predictions from {pred_file}")
    data = np.load(pred_file, allow_pickle=True)
    
    y_true = data['y_true']
    probs = data['probs']
    
    print(f"  y_true shape: {y_true.shape}")
    print(f"  probs shape: {probs.shape}")
    print(f"  Accuracy: {np.mean(y_true == (probs >= 0.5)):.4f}")
    
    # Compute all metrics
    results = {
        "metadata": {
            "n_samples": int(len(y_true)),
            "pred_file": str(pred_file),
        },
        "metrics": {}
    }
    
    # 1. Brier Score
    brier = compute_brier_score(y_true, probs)
    results["metrics"]["brier_score"] = {
        "value": brier,
        "description": "Mean squared error between predicted and empirical probabilities. Lower is better."
    }
    print(f"  Brier Score: {brier:.6f}")
    
    # 2. Negative Log-Likelihood
    nll = compute_nll(y_true, probs)
    results["metrics"]["nll"] = {
        "value": nll,
        "description": "Negative log-likelihood (cross-entropy). Lower is better."
    }
    print(f"  NLL: {nll:.6f}")
    
    # 3. Adaptive ECE (equal-mass)
    adaptive_ece, adaptive_bins = compute_ece_adaptive(y_true, probs, n_bins=10)
    results["metrics"]["ece_adaptive"] = {
        "value": adaptive_ece,
        "n_bins": 10,
        "binning": "equal-mass (equal-count)",
        "description": "Adaptive ECE using equal-mass bins. Robust to skewed confidence distributions.",
        "bins": adaptive_bins
    }
    print(f"  Adaptive ECE (10 equal-mass bins): {adaptive_ece:.6f}")
    
    # 4. ECE with multiple bin counts
    for n_bins in [10, 15, 20]:
        ece_val, bin_info = compute_ece_equal_width(y_true, probs, n_bins=n_bins)
        results["metrics"][f"ece_equal_width_bins_{n_bins}"] = {
            "value": ece_val,
            "n_bins": n_bins,
            "binning": "equal-width",
            "description": "Standard ECE with equal-width confidence bins.",
            "bins": bin_info
        }
        print(f"  ECE (equal-width, {n_bins} bins): {ece_val:.6f}")
    
    # Include standard ECE (10 bins) for comparison
    standard_ece = results["metrics"]["ece_equal_width_bins_10"]["value"]
    results["metrics"]["ece_standard"] = {
        "value": standard_ece,
        "description": "Standard ECE (10 equal-width bins) - same as paper baseline"
    }

    # Table XVI consistency check values.
    targets = {
        "ece_equal_width_bins_10": 0.1076,
        "ece_equal_width_bins_15": 0.1068,
        "ece_equal_width_bins_20": 0.1065,
        "ece_adaptive": 0.1109,
    }
    checks = {}
    for key, target in targets.items():
        actual = float(results["metrics"][key]["value"])
        checks[key] = {
            "target": target,
            "actual": actual,
            "abs_diff": abs(actual - target),
        }
    results["table_xvi_check"] = checks
    
    return results


def main():
    """Main: compute metrics and save results."""
    
    # Paths
    pred_file = Path("artifacts/preds/CalibraTeach.npz")
    output_dir = Path("artifacts")
    output_file = output_dir / "calibration_robustness_metrics.json"
    
    if not pred_file.exists():
        print(f"ERROR: Prediction file not found: {pred_file}")
        return
    
    # Compute metrics
    print("\n" + "="*70)
    print("COMPUTING ADVANCED CALIBRATION METRICS")
    print("="*70 + "\n")
    
    results = compute_calibration_metrics(pred_file)
    
    # Save results
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_file}")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Calibration Metrics Comparison")
    print("="*70)
    print(f"{'Metric':<30} {'Value':<15} {'Notes'}")
    print("-" * 70)
    
    bs = results["metrics"]["brier_score"]["value"]
    nll = results["metrics"]["nll"]["value"]
    ece_adaptive = results["metrics"]["ece_adaptive"]["value"]
    ece_10 = results["metrics"]["ece_equal_width_bins_10"]["value"]
    ece_15 = results["metrics"]["ece_equal_width_bins_15"]["value"]
    ece_20 = results["metrics"]["ece_equal_width_bins_20"]["value"]
    
    print(f"{'Brier Score':<30} {bs:<15.6f} {'(0-1 range, lower better)'}")
    print(f"{'NLL (Cross-Entropy)':<30} {nll:<15.6f} {'(lower better)'}")
    print(f"{'ECE (10 equal-width)':<30} {ece_10:<15.6f} {'(baseline from paper)'}")
    print(f"{'ECE (15 equal-width)':<30} {ece_15:<15.6f} {'(stability check)'}")
    print(f"{'ECE (20 equal-width)':<30} {ece_20:<15.6f} {'(stability check)'}")
    print(f"{'ECE (10 equal-mass/adaptive)':<30} {ece_adaptive:<15.6f} {'(robust to skew)'}")
    
    print("\n✓ All metrics computed successfully")
    return results


if __name__ == "__main__":
    main()
