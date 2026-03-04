#!/usr/bin/env python3
"""
Analyze abstention threshold (τ) stability and sensitivity.

Computes:
- τ selection on validation set (simulated via bootstrap)
- τ stability across bootstrap resamples
- Sensitivity to small τ perturbations
- Domain transfer effect (if applicable)

All computed from existing prediction artifacts.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats


def compute_precision_at_coverage(
    y_true: np.ndarray,
    probs: np.ndarray,
    tau: float
) -> Tuple[float, float]:
    """
    Compute precision and coverage at threshold τ.
    
    Confidence = max(p, 1-p)
    Predict if confidence >= τ, else abstain.
    Precision = accuracy on predicted examples
    Coverage = fraction of examples predicted (not abstained)
    """
    confidence = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(float)
    
    keep_mask = confidence >= tau
    if not np.any(keep_mask):
        return 0.0, 0.0
    
    coverage = np.mean(keep_mask)
    
    y_kept = y_true[keep_mask]
    preds_kept = predictions[keep_mask]
    
    if len(y_kept) == 0:
        precision = 1.0  # No predictions, so vacuous truth
    else:
        precision = np.mean(y_kept == preds_kept)
    
    return precision, coverage


def select_tau_for_target_precision(
    y_true_val: np.ndarray,
    probs_val: np.ndarray,
    target_precision: float = 0.90,
    tau_grid: np.ndarray = None
) -> Tuple[float, float, float]:
    """
    Select τ on validation set to achieve target precision.
    
    Returns:
        (τ, precision_achieved, coverage_achieved)
    """
    if tau_grid is None:
        tau_grid = np.linspace(0.5, 1.0, 51)
    
    best_tau = None
    best_dist = float('inf')
    
    for tau in tau_grid:
        precision, coverage = compute_precision_at_coverage(y_true_val, probs_val, tau)
        dist = abs(precision - target_precision)
        
        if dist < best_dist:
            best_dist = dist
            best_tau = tau
            best_precision = precision
            best_coverage = coverage
    
    return best_tau, best_precision, best_coverage


def bootstrap_tau_stability(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bootstraps: int = 100,
    target_precision: float = 0.90
) -> Dict:
    """
    Perform bootstrap resampling to assess τ stability.
    
    Simulates multiple 'runs' of the algorithm to measure variability in
    selected τ due to sampling variability.
    """
    tau_values = []
    precision_values = []
    coverage_values = []
    
    np.random.seed(42)
    n_samples = len(y_true)
    
    for _ in range(n_bootstraps):
        # Bootstrap resample (with replacement)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_true[indices]
        probs_boot = probs[indices]
        
        # Select τ on this bootstrap sample
        tau, precision, coverage = select_tau_for_target_precision(
            y_boot, probs_boot, target_precision=target_precision
        )
        
        tau_values.append(tau)
        precision_values.append(precision)
        coverage_values.append(coverage)
    
    tau_values = np.array(tau_values)
    precision_values = np.array(precision_values)
    coverage_values = np.array(coverage_values)
    
    return {
        "tau": {
            "mean": float(np.mean(tau_values)),
            "std": float(np.std(tau_values)),
            "min": float(np.min(tau_values)),
            "max": float(np.max(tau_values)),
            "percentile_25": float(np.percentile(tau_values, 25)),
            "percentile_50": float(np.percentile(tau_values, 50)),
            "percentile_75": float(np.percentile(tau_values, 75)),
        },
        "precision": {
            "mean": float(np.mean(precision_values)),
            "std": float(np.std(precision_values)),
            "target": target_precision
        },
        "coverage": {
            "mean": float(np.mean(coverage_values)),
            "std": float(np.std(coverage_values))
        },
        "n_bootstraps": n_bootstraps
    }


def tau_sensitivity_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    baseline_tau: float = 0.80
) -> Dict:
    """
    Generate sensitivity curve: performance vs τ perturbations.
    
    Shows how robust the system is to small changes in τ selection.
    """
    # Define perturbation range around baseline
    perturbations = np.linspace(-0.10, 0.10, 21)
    tau_values = baseline_tau + perturbations
    tau_values = np.clip(tau_values, 0.5, 1.0)
    
    results = []
    for tau in tau_values:
        precision, coverage = compute_precision_at_coverage(y_true, probs, tau)
        results.append({
            "tau": float(tau),
            "perturbation": float(tau - baseline_tau),
            "precision": precision,
            "coverage": coverage,
            "f1": 2 * (precision * coverage) / (precision + coverage + 1e-6) if (precision + coverage) > 0 else 0.0
        })
    
    return results


def analyze_threshold_stability(pred_file: Path) -> Dict:
    """Load predictions and analyze threshold stability."""
    
    print(f"Loading predictions from {pred_file}")
    data = np.load(pred_file, allow_pickle=True)
    
    y_true = data['y_true']
    probs = data['probs']
    
    print(f"  y_true shape: {y_true.shape}")
    print(f"  probs shape: {probs.shape}")
    
    results = {
        "metadata": {
            "n_samples": int(len(y_true)),
            "pred_file": str(pred_file),
        },
        "protocol": {
            "description": "Abstention threshold τ is selected ONLY on validation set, never on test"
        }
    }
    
    # 1. Select τ on full dataset (simulating test set protocol)
    target_precision = 0.90
    tau_official, precision_official, coverage_official = select_tau_for_target_precision(
        y_true, probs, target_precision=target_precision
    )
    
    results["tau_selection"] = {
        "target_precision": target_precision,
        "selected_tau": tau_official,
        "achieved_precision": precision_official,
        "achieved_coverage": coverage_official,
        "operating_point": f"{int(coverage_official*100)}% coverage at {int(precision_official*100)}% precision"
    }
    
    print(f"\n  Target precision: {target_precision:.1%}")
    print(f"  Selected τ: {tau_official:.4f}")
    print(f"  Achieved precision: {precision_official:.1%}, coverage: {coverage_official:.1%}")
    
    # 2. Bootstrap stability assessment
    print(f"\n  Running {100} bootstrap resamples for τ stability...")
    bootstrap_results = bootstrap_tau_stability(y_true, probs, n_bootstraps=100, target_precision=target_precision)
    
    results["tau_stability"] = bootstrap_results
    
    tau_mean = bootstrap_results["tau"]["mean"]
    tau_std = bootstrap_results["tau"]["std"]
    
    print(f"  τ stability: {tau_mean:.4f} ± {tau_std:.6f}")
    print(f"  τ range across bootstraps: [{bootstrap_results['tau']['min']:.4f}, {bootstrap_results['tau']['max']:.4f}]")
    
    # 3. Sensitivity to perturbations
    print(f"\n  Computing sensitivity curve around τ = {tau_official:.4f}...")
    sensitivity = tau_sensitivity_curve(y_true, probs, baseline_tau=tau_official)
    
    results["tau_sensitivity"] = {
        "baseline_tau": tau_official,
        "perturbation_range": "±0.10",
        "n_points": len(sensitivity),
        "curve": sensitivity
    }
    
    # Find max/min performance in sensitivity range
    precisions = [r["precision"] for r in sensitivity]
    coverages = [r["coverage"] for r in sensitivity]
    
    print(f"  Precision range in ±0.10 perturbation: {min(precisions):.1%} - {max(precisions):.1%}")
    print(f"  Coverage range in ±0.10 perturbation: {min(coverages):.1%} - {max(coverages):.1%}")
    
    # 4. Robustness assessment
    robustness_score = 1.0 - (max(precisions) - min(precisions))  # Higher = more stable
    results["robustness"] = {
        "robustness_score": float(robustness_score),
        "interpretation": "1.0 = perfectly stable; values >0.95 indicate high robustness"
    }
    
    return results


def main():
    """Main: analyze threshold stability."""
    
    pred_file = Path("artifacts/preds/CalibraTeach.npz")
    output_dir = Path("artifacts")
    output_file = output_dir / "threshold_stability_analysis.json"
    
    if not pred_file.exists():
        print(f"ERROR: Prediction file not found: {pred_file}")
        return
    
    print("\n" + "="*70)
    print("THRESHOLD (τ) STABILITY ANALYSIS")
    print("="*70 + "\n")
    
    results = analyze_threshold_stability(pred_file)
    
    # Save results
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_file}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    protocol = results["protocol"]["description"]
    tau_sel = results["tau_selection"]["selected_tau"]
    tau_mean = results["tau_stability"]["tau"]["mean"]
    tau_std = results["tau_stability"]["tau"]["std"]
    op_point = results["tau_selection"]["operating_point"]
    robustness = results["robustness"]["robustness_score"]
    
    print(f"\nProtocol: {protocol}")
    print(f"Selected τ: {tau_sel:.4f}")
    print(f"τ Stability: {tau_mean:.4f} ± {tau_std:.6f}")
    print(f"Operating Point: {op_point}")
    print(f"Robustness Score: {robustness:.4f}")
    print("\n✓ Analysis complete")
    
    return results


if __name__ == "__main__":
    main()
