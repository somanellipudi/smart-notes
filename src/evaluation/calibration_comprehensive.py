"""
Enhanced Calibration Evaluation Module.

Generates comprehensive calibration analysis including:
- Reliability diagrams before/after calibration
- ECE at multiple bin sizes
- Temperature scaling parameter estimation
- Calibration comparison tables

This module extends the base calibration.py with full reporting capabilities.
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from scipy.optimize import minimize
from src.evaluation.calibration import CalibrationEvaluator

logger = logging.getLogger(__name__)


@dataclass
class CalibrationReport:
    """Complete calibration evaluation report."""
    ece_before: Dict[int, float]  # n_bins -> ECE before calibration
    ece_after: Dict[int, float]  # n_bins -> ECE after calibration
    temperature: float  # Optimal temperature parameter
    brier_before: float
    brier_after: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert ECE bin comparison to DataFrame."""
        rows = []
        for n_bins in sorted(self.ece_before.keys()):
            rows.append({
                "Bins": n_bins,
                "ECE Before": self.ece_before[n_bins],
                "ECE After": self.ece_after.get(n_bins, 0.0),
                "Improvement": self.ece_before[n_bins] - self.ece_after.get(n_bins, 0.0)
            })
        return pd.DataFrame(rows)


def temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to logits.
    
    Args:
        logits: Raw logits from model, shape (n_samples, n_classes)
        temperature: Temperature parameter (T > 1 softens, T < 1 sharpens)
    
    Returns:
        Calibrated probabilities
    """
    scaled_logits = logits / temperature
    
    # Apply softmax
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    return probs


def temperature_scale_confidences(
    confidences: np.ndarray,
    temperature: float
) -> np.ndarray:
    """
    Apply temperature scaling to confidence scores.
    
    For binary or pre-computed confidences, we approximate temperature scaling
    by transforming the confidence space.
    
    Args:
        confidences: Confidence scores in [0, 1]
        temperature: Temperature parameter
    
    Returns:
        Calibrated confidence scores
    """
    # Convert confidences to logits (inverse sigmoid)
    epsilon = 1e-10
    confidences = np.clip(confidences, epsilon, 1 - epsilon)
    logits = np.log(confidences / (1 - confidences))
    
    # Scale logits
    scaled_logits = logits / temperature
    
    # Convert back to probabilities
    calibrated = 1 / (1 + np.exp(-scaled_logits))
    
    return calibrated


def find_optimal_temperature(
    confidences: np.ndarray,
    correctness: np.ndarray,
    init_temp: float = 1.5
) -> float:
    """
    Find optimal temperature parameter via NLL minimization.
    
    Args:
        confidences: Model confidence scores
        correctness: Binary correctness labels (1=correct, 0=incorrect)
        init_temp: Initial temperature guess
    
    Returns:
        Optimal temperature parameter
    """
    
    def nll_loss(temp):
        """Negative log-likelihood loss."""
        calibrated = temperature_scale_confidences(confidences, temp)
        
        # Clip to avoid log(0)
        epsilon = 1e-10
        calibrated = np.clip(calibrated, epsilon, 1 - epsilon)
        
        # NLL for binary classification
        nll = -np.mean(
            correctness * np.log(calibrated) +
            (1 - correctness) * np.log(1 - calibrated)
        )
        return nll
    
    # Optimize temperature (must be positive)
    result = minimize(
        nll_loss,
        x0=init_temp,
        method='L-BFGS-B',
        bounds=[(0.1, 10.0)]
    )
    
    optimal_temp = result.x[0]
    logger.info(f"Optimal temperature: {optimal_temp:.4f}")
    
    return optimal_temp


def evaluate_calibration_comprehensive(
    confidences: List[float],
    correctness: List[int],
    bin_sizes: List[int] = [10, 15, 20],
    output_dir: Optional[Path] = None
) -> CalibrationReport:
    """
    Comprehensive calibration evaluation before and after temperature scaling.
    
    Args:
        confidences: Model confidence scores
        correctness: Binary correctness (1=correct, 0=incorrect)
        bin_sizes: List of bin sizes to evaluate ECE
        output_dir: Directory to save outputs
    
    Returns:
        CalibrationReport
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    logger.info("Computing calibration metrics...")
    
    # Evaluate BEFORE calibration
    ece_before = {}
    for n_bins in bin_sizes:
        evaluator = CalibrationEvaluator(n_bins=n_bins)
        ece = evaluator.expected_calibration_error(confidences, correctness)
        ece_before[n_bins] = ece
        logger.info(f"ECE before ({n_bins} bins): {ece:.4f}")
    
    evaluator_default = CalibrationEvaluator(n_bins=15)
    brier_before = evaluator_default.brier_score(confidences, correctness)
    
    # Find optimal temperature
    logger.info("Finding optimal temperature...")
    temperature = find_optimal_temperature(confidences, correctness)
    
    # Apply temperature scaling
    calibrated_confidences = temperature_scale_confidences(confidences, temperature)
    
    # Evaluate AFTER calibration
    ece_after = {}
    for n_bins in bin_sizes:
        evaluator = CalibrationEvaluator(n_bins=n_bins)
        ece = evaluator.expected_calibration_error(calibrated_confidences, correctness)
        ece_after[n_bins] = ece
        logger.info(f"ECE after ({n_bins} bins): {ece:.4f}")
    
    brier_after = evaluator_default.brier_score(calibrated_confidences, correctness)
    
    # Generate reliability diagrams
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Before calibration
        plot_reliability_diagram(
            confidences,
            correctness,
            n_bins=15,
            title="Reliability Diagram (Before Calibration)",
            save_path=output_dir / "fig_reliability_before.png"
        )
        
        # After calibration
        plot_reliability_diagram(
            calibrated_confidences,
            correctness,
            n_bins=15,
            title="Reliability Diagram (After Temperature Scaling)",
            save_path=output_dir / "fig_reliability_after.png"
        )
        
        logger.info(f"Reliability diagrams saved to {output_dir}")
    
    # Create report
    report = CalibrationReport(
        ece_before=ece_before,
        ece_after=ece_after,
        temperature=temperature,
        brier_before=brier_before,
        brier_after=brier_after
    )
    
    # Save tables
    if output_dir:
        df = report.to_dataframe()
        df.to_csv(output_dir / "ece_bins_table.csv", index=False)
        
        # Save as markdown
        with open(output_dir / "ece_bins_table.md", 'w') as f:
            f.write(df.to_markdown(index=False, floatfmt=".4f"))
        
        # Save JSON report
        with open(output_dir / "calibration_report.json", 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Calibration tables saved to {output_dir}")
    
    return report


def plot_reliability_diagram(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: Optional[Path] = None
):
    """
    Plot reliability diagram showing calibration quality.
    
    Args:
        confidences: Confidence scores
        correctness: Binary correctness labels
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(correctness[in_bin])
            count = np.sum(in_bin)
            
            bin_confidences.append(avg_confidence)
            bin_accuracies.append(avg_accuracy)
            bin_counts.append(count)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Plot actual calibration
    if bin_confidences:
        ax.plot(bin_confidences, bin_accuracies, 'o-', label='Model', 
                markersize=8, linewidth=2)
        
        # Add bin counts as text
        for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
            ax.annotate(f'{count}', (conf, acc), 
                       textcoords="offset points", 
                       xytext=(0, 5), 
                       ha='center',
                       fontsize=8,
                       alpha=0.7)
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Reliability diagram saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n_samples = 500
    
    # Simulate overconfident model
    true_probs = np.random.uniform(0.5, 0.9, n_samples)
    confidences = true_probs + 0.1  # Overconfident
    confidences = np.clip(confidences, 0, 1)
    
    # Generate correctness
    correctness = (np.random.uniform(0, 1, n_samples) < true_probs).astype(int)
    
    # Evaluate calibration
    report = evaluate_calibration_comprehensive(
        confidences=confidences.tolist(),
        correctness=correctness.tolist(),
        bin_sizes=[10, 15, 20],
        output_dir=Path("artifacts/calibration_test")
    )
    
    print("\nCalibration Report:")
    print(report.to_dataframe())
    print(f"\nOptimal Temperature: {report.temperature:.4f}")
    print(f"Brier Before: {report.brier_before:.4f}")
    print(f"Brier After: {report.brier_after:.4f}")
