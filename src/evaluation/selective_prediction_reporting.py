"""
Selective Prediction Reporting Module.

Generates comprehensive selective prediction analysis:
- Accuracy at different coverage levels
- Coverage at different risk thresholds
- Risk-coverage curves
- Recommended operating points for deployment

This extends the base selective_prediction.py with full reporting.
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SelectivePredictionReport:
    """Complete selective prediction report."""
    accuracy_at_coverage: Dict[float, float]  # coverage -> accuracy
    coverage_at_risk: Dict[float, float]  # max_risk -> coverage
    auc_rc: float  # Area under risk-coverage curve
    optimal_operating_point: Dict[str, float]  # threshold, coverage, accuracy, risk
    
    def to_dict(self) -> Dict:
        return asdict(self)


def compute_accuracy_at_coverage(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_coverages: List[float] = [1.0, 0.9, 0.8]
) -> Dict[float, float]:
    """
    Compute accuracy at specified coverage levels.
    
    Coverage is achieved by rejecting low-confidence predictions.
    
    Args:
        confidences: Confidence scores
        predictions: Predicted labels
        targets: True labels
        target_coverages: Desired coverage levels (fraction of predictions to keep)
    
    Returns:
        Dict mapping coverage -> accuracy
    """
    n_samples = len(confidences)
    
    # Sort by confidence (descending)
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    results = {}
    
    for coverage in target_coverages:
        n_keep = int(n_samples * coverage)
        
        if n_keep == 0:
            results[coverage] = 0.0
            continue
        
        # Take top n_keep most confident predictions
        kept_predictions = sorted_predictions[:n_keep]
        kept_targets = sorted_targets[:n_keep]
        
        accuracy = np.mean(kept_predictions == kept_targets)
        results[coverage] = float(accuracy)
        
        logger.info(f"Coverage {coverage:.0%}: Accuracy = {accuracy:.4f} (n={n_keep})")
    
    return results


def compute_coverage_at_risk(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_risks: List[float] = [0.10, 0.05]
) -> Dict[float, float]:
    """
    Compute coverage at specified risk thresholds.
    
    Risk is the maximum error rate allowed on accepted predictions.
    We find the highest coverage that maintains risk below threshold.
    
    Args:
        confidences: Confidence scores
        predictions: Predicted labels
        targets: True labels
        target_risks: Maximum acceptable risk (error rate) thresholds
    
    Returns:
        Dict mapping risk_threshold -> coverage
    """
    n_samples = len(confidences)
    
    # Sort by confidence (descending)
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    results = {}
    
    for max_risk in target_risks:
        # Find maximum coverage while keeping risk <= max_risk
        best_coverage = 0.0
        
        for n_keep in range(1, n_samples + 1):
            kept_predictions = sorted_predictions[:n_keep]
            kept_targets = sorted_targets[:n_keep]
            
            # Compute risk (error rate)
            errors = np.sum(kept_predictions != kept_targets)
            risk = errors / n_keep
            
            if risk <= max_risk:
                best_coverage = n_keep / n_samples
            else:
                break  # Risk exceeds threshold, stop
        
        results[max_risk] = float(best_coverage)
        logger.info(f"Max risk {max_risk:.1%}: Coverage = {best_coverage:.4f}")
    
    return results


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute full risk-coverage curve.
    
    Args:
        confidences: Confidence scores
        predictions: Predicted labels
        targets: True labels
        num_points: Number of points on curve
    
    Returns:
        Tuple of (coverages, risks, auc_rc)
    """
    n_samples = len(confidences)
    
    # Sort by confidence (descending)
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    coverages = []
    risks = []
    
    # Compute risk at each coverage level
    for n_keep in range(1, n_samples + 1):
        kept_predictions = sorted_predictions[:n_keep]
        kept_targets = sorted_targets[:n_keep]
        
        coverage = n_keep / n_samples
        errors = np.sum(kept_predictions != kept_targets)
        risk = errors / n_keep
        
        coverages.append(coverage)
        risks.append(risk)
    
    coverages = np.array(coverages)
    risks = np.array(risks)
    
    # Compute AUC-RC (lower is better)
    auc_rc = np.trapz(risks, coverages)
    
    return coverages, risks, auc_rc


def find_optimal_operating_point(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_coverage: float = 0.9
) -> Dict[str, float]:
    """
    Find optimal confidence threshold for a target coverage.
    
    Args:
        confidences: Confidence scores
        predictions: Predicted labels
        targets: True labels
        target_coverage: Desired coverage (e.g., 0.9 for 90%)
    
    Returns:
        Dict with threshold, coverage, accuracy, risk
    """
    n_samples = len(confidences)
    n_keep = int(n_samples * target_coverage)
    
    # Sort by confidence (descending)
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_confidences = confidences[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    # Find threshold
    if n_keep > 0 and n_keep <= n_samples:
        threshold = sorted_confidences[n_keep - 1]
        
        # Compute metrics at this threshold
        kept_predictions = sorted_predictions[:n_keep]
        kept_targets = sorted_targets[:n_keep]
        
        accuracy = np.mean(kept_predictions == kept_targets)
        errors = np.sum(kept_predictions != kept_targets)
        risk = errors / n_keep
        actual_coverage = n_keep / n_samples
        
    else:
        threshold = 0.0
        accuracy = 0.0
        risk = 1.0
        actual_coverage = 0.0
    
    return {
        "threshold": float(threshold),
        "coverage": float(actual_coverage),
        "accuracy": float(accuracy),
        "risk": float(risk)
    }


def generate_selective_prediction_report(
    confidences: List[float],
    predictions: List[int],
    targets: List[int],
    target_coverages: List[float] = [1.0, 0.9, 0.8],
    target_risks: List[float] = [0.10, 0.05],
    output_dir: Optional[Path] = None
) -> SelectivePredictionReport:
    """
    Generate comprehensive selective prediction report.
    
    Args:
        confidences: Confidence scores
        predictions: Predicted labels
        targets: True labels
        target_coverages: Coverage levels to evaluate
        target_risks: Risk thresholds to evaluate
        output_dir: Directory to save outputs
    
    Returns:
        SelectivePredictionReport
    """
    confidences = np.array(confidences)
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    logger.info("Computing selective prediction metrics...")
    
    # Accuracy at coverage
    acc_at_cov = compute_accuracy_at_coverage(
        confidences, predictions, targets, target_coverages
    )
    
    # Coverage at risk
    cov_at_risk = compute_coverage_at_risk(
        confidences, predictions, targets, target_risks
    )
    
    # Risk-coverage curve
    coverages, risks, auc_rc = compute_risk_coverage_curve(
        confidences, predictions, targets
    )
    
    # Optimal operating point (90% coverage)
    optimal_op = find_optimal_operating_point(
        confidences, predictions, targets, target_coverage=0.9
    )
    
    # Create report
    report = SelectivePredictionReport(
        accuracy_at_coverage=acc_at_cov,
        coverage_at_risk=cov_at_risk,
        auc_rc=auc_rc,
        optimal_operating_point=optimal_op
    )
    
    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save accuracy at coverage table
        df_acc = pd.DataFrame([
            {"Coverage": f"{cov:.0%}", "Accuracy": acc}
            for cov, acc in acc_at_cov.items()
        ])
        df_acc.to_csv(output_dir / "selective_accuracy_at_coverage.csv", index=False)
        with open(output_dir / "selective_accuracy_at_coverage.md", 'w') as f:
            f.write(df_acc.to_markdown(index=False, floatfmt=".4f"))
        
        # Save coverage at risk table
        df_cov = pd.DataFrame([
            {"Max Risk": f"{risk:.1%}", "Coverage": f"{cov:.2%}"}
            for risk, cov in cov_at_risk.items()
        ])
        df_cov.to_csv(output_dir / "selective_coverage_at_risk.csv", index=False)
        with open(output_dir / "selective_coverage_at_risk.md", 'w') as f:
            f.write(df_cov.to_markdown(index=False))
        
        # Save risk-coverage curve plot
        plot_risk_coverage_curve(
            coverages, risks, auc_rc,
            save_path=output_dir / "fig_risk_coverage.png"
        )
        
        # Save JSON report
        with open(output_dir / "selective_prediction_report.json", 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Selective prediction reports saved to {output_dir}")
    
    return report


def plot_risk_coverage_curve(
    coverages: np.ndarray,
    risks: np.ndarray,
    auc_rc: float,
    save_path: Optional[Path] = None
):
    """
    Plot risk-coverage curve.
    
    Args:
        coverages: Coverage values
        risks: Risk (error rate) values
        auc_rc: Area under curve
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(coverages, risks, linewidth=2, label=f'Risk-Coverage Curve (AUC={auc_rc:.4f})')
    ax.fill_between(coverages, risks, alpha=0.3)
    
    # Mark common operating points
    for target_cov in [0.8, 0.9, 1.0]:
        idx = np.argmin(np.abs(coverages - target_cov))
        if idx < len(coverages):
            ax.plot(coverages[idx], risks[idx], 'ro', markersize=8)
            ax.annotate(
                f'{target_cov:.0%} coverage\nRisk: {risks[idx]:.2%}',
                (coverages[idx], risks[idx]),
                textcoords="offset points",
                xytext=(10, -10),
                ha='left',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)
            )
    
    ax.set_xlabel('Coverage (Fraction of Predictions Accepted)', fontsize=12)
    ax.set_ylabel('Risk (Error Rate on Accepted Predictions)', fontsize=12)
    ax.set_title('Selective Prediction: Risk-Coverage Tradeoff', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(risks) * 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Risk-coverage curve saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n_samples = 500
    
    # Simulate predictions with varying confidence
    confidences = np.random.uniform(0.5, 1.0, n_samples)
    predictions = np.random.randint(0, 3, n_samples)
    
    # Generate targets (high confidence = more likely correct)
    prob_correct = (confidences - 0.5) / 0.5  # Linear relationship
    is_correct = np.random.uniform(0, 1, n_samples) < prob_correct
    targets = np.where(is_correct, predictions, (predictions + 1) % 3)
    
    # Generate report
    report = generate_selective_prediction_report(
        confidences=confidences.tolist(),
        predictions=predictions.tolist(),
        targets=targets.tolist(),
        output_dir=Path("artifacts/selective_test")
    )
    
    print("\nSelective Prediction Report:")
    print(f"AUC-RC: {report.auc_rc:.4f}")
    print(f"\nOptimal Operating Point (90% coverage):")
    print(f"  Threshold: {report.optimal_operating_point['threshold']:.4f}")
    print(f"  Accuracy: {report.optimal_operating_point['accuracy']:.4f}")
    print(f"  Risk: {report.optimal_operating_point['risk']:.4f}")
