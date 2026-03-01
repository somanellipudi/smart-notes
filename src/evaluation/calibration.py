"""
Calibration evaluation for confidence scores.

Metrics:
- Expected Calibration Error (ECE)
- Brier Score
- Reliability diagrams
- Confidence histograms

Determines if model confidence correlates with actual accuracy.
"""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


class CalibrationEvaluator:
    """
    Evaluate calibration of confidence scores.
    
    A well-calibrated model should have:
    - ECE close to 0
    - Brier score close to 0
    - Reliability diagram on diagonal
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize calibration evaluator.
        
        Args:
            n_bins: Number of bins for reliability diagram
        """
        self.n_bins = n_bins
    
    def evaluate(
        self,
        predictions: List[float],
        labels: List[int],
        return_bins: bool = False
    ) -> Dict[str, float]:
        """
        Compute calibration metrics.
        
        Args:
            predictions: Confidence scores in [0, 1]
            labels: Binary labels (1=correct, 0=incorrect)
            return_bins: Whether to return bin statistics
        
        Returns:
            Dict with ECE, Brier score, and optional bin stats
        """
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Expected Calibration Error
        ece = self.expected_calibration_error(predictions, labels)
        
        # Brier Score
        brier = self.brier_score(predictions, labels)
        
        # Accuracy
        threshold = 0.5
        pred_labels = (predictions >= threshold).astype(int)
        accuracy = np.mean(pred_labels == labels)
        
        metrics = {
            "ece": ece,
            "brier_score": brier,
            "accuracy": accuracy,
            "n_samples": len(predictions)
        }
        
        if return_bins:
            bin_stats = self._compute_bin_statistics(predictions, labels)
            metrics["bins"] = bin_stats
        
        logger.info(
            f"Calibration: ECE={ece:.4f}, Brier={brier:.4f}, Acc={accuracy:.2%}"
        )
        
        return metrics
    
    def expected_calibration_error(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures the difference between confidence and accuracy
        across bins.
        
        Args:
            predictions: Confidence scores
            labels: Ground truth labels
        
        Returns:
            ECE value (lower is better)
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(predictions)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin])
                avg_confidence_in_bin = np.mean(predictions[in_bin])
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def brier_score(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute Brier score (mean squared error of probabilities).
        
        Args:
            predictions: Confidence scores
            labels: Ground truth labels
        
        Returns:
            Brier score (lower is better)
        """
        return float(np.mean((predictions - labels) ** 2))
    
    def plot_reliability_diagram(
        self,
        predictions: List[float],
        labels: List[int],
        save_path: Optional[str] = None,
        title: str = "Reliability Diagram"
    ):
        """
        Plot reliability diagram (calibration curve).
        
        Args:
            predictions: Confidence scores
            labels: Ground truth labels
            save_path: Path to save plot
            title: Plot title
        """
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        bin_stats = self._compute_bin_statistics(predictions, labels)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Reliability diagram
        confidences = []
        accuracies = []
        counts = []
        
        for stats in bin_stats:
            if stats["count"] > 0:
                confidences.append(stats["avg_confidence"])
                accuracies.append(stats["accuracy"])
                counts.append(stats["count"])
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.plot(confidences, accuracies, 'ro-', label='Model calibration', linewidth=2, markersize=8)
        
        # Add bar widths proportional to counts
        max_count = max(counts) if counts else 1
        for conf, acc, count in zip(confidences, accuracies, counts):
            width = 0.05 * (count / max_count)
            ax1.bar(conf, acc, width=width, alpha=0.3, color='blue')
        
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Confidence histogram
        ax2.hist(predictions, bins=self.n_bins, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Reliability diagram saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _compute_bin_statistics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> List[Dict]:
        """Compute statistics for each confidence bin."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_stats = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            count = np.sum(in_bin)
            
            if count > 0:
                accuracy = np.mean(labels[in_bin])
                avg_confidence = np.mean(predictions[in_bin])
                gap = np.abs(avg_confidence - accuracy)
            else:
                accuracy = 0.0
                avg_confidence = (bin_lower + bin_upper) / 2
                gap = 0.0
            
            bin_stats.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": int(count),
                "accuracy": float(accuracy),
                "avg_confidence": float(avg_confidence),
                "gap": float(gap)
            })
        
        return bin_stats
    
    def compute_sharpness(self, predictions: List[float]) -> float:
        """
        Compute sharpness (variance of predictions).
        
        Higher sharpness means model is more decisive.
        
        Args:
            predictions: Confidence scores
        
        Returns:
            Sharpness value
        """
        return float(np.var(predictions))

    # -----------------
    # Temperature scaling
    # -----------------
    def _apply_temperature(self, probs: np.ndarray, tau: float) -> np.ndarray:
        """
        Apply temperature scaling to probabilities via logits.
        If inputs are probabilities p in (0,1), we compute logits = logit(p)
        and return sigmoid(logits / tau).
        """
        eps = 1e-12
        p = np.clip(probs, eps, 1 - eps)
        logits = np.log(p / (1 - p))
        scaled = 1.0 / (1.0 + np.exp(-logits / float(tau)))
        return scaled

    def fit_temperature_grid(
        self,
        val_probs: List[float],
        val_labels: List[int],
        grid_min: float = 0.8,
        grid_max: float = 2.0,
        grid_steps: int = 100
    ) -> Dict[str, float]:
        """
        Fit a temperature parameter on validation probabilities by minimizing ECE.

        Args:
            val_probs: Validation predicted probabilities (or correctness proxies)
            val_labels: Validation binary labels (1=correct, 0=incorrect)
            grid_min: minimum tau
            grid_max: maximum tau
            grid_steps: number of grid points

        Returns:
            Dict with keys: best_tau, best_ece, ece_grid (list)
        """
        probs = np.array(val_probs)
        labels = np.array(val_labels)

        taus = np.linspace(grid_min, grid_max, grid_steps)
        ece_values = []
        for tau in taus:
            scaled = self._apply_temperature(probs, tau)
            ece = self.expected_calibration_error(scaled, labels)
            ece_values.append(float(ece))

        best_idx = int(np.argmin(ece_values))
        best_tau = float(taus[best_idx])
        best_ece = float(ece_values[best_idx])

        return {"best_tau": best_tau, "best_ece": best_ece, "taus": taus.tolist(), "ece_grid": ece_values}
    
    def plot_confidence_by_correctness(
        self,
        predictions: List[float],
        labels: List[int],
        save_path: Optional[str] = None
    ):
        """
        Plot confidence distributions for correct vs incorrect predictions.
        
        Args:
            predictions: Confidence scores
            labels: Ground truth labels
            save_path: Path to save plot
        """
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Separate by correctness (assuming threshold=0.5)
        pred_labels = (predictions >= 0.5).astype(int)
        correct_mask = (pred_labels == labels)
        
        correct_conf = predictions[correct_mask]
        incorrect_conf = predictions[~correct_mask]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(correct_conf, bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
        ax.hist(incorrect_conf, bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
        
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Confidence by Correctness', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add mean lines
        if len(correct_conf) > 0:
            ax.axvline(np.mean(correct_conf), color='green', linestyle='--', 
                      label=f'Correct mean: {np.mean(correct_conf):.3f}')
        if len(incorrect_conf) > 0:
            ax.axvline(np.mean(incorrect_conf), color='red', linestyle='--',
                      label=f'Incorrect mean: {np.mean(incorrect_conf):.3f}')
        
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confidence plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def evaluate_calibration_from_file(
    predictions_file: str,
    labels_file: str,
    output_dir: str = "outputs/calibration"
) -> Dict[str, float]:
    """
    Load predictions and labels from files and evaluate calibration.
    
    Args:
        predictions_file: Path to predictions (one per line)
        labels_file: Path to labels (one per line)
        output_dir: Directory to save plots
    
    Returns:
        Calibration metrics
    """
    predictions = np.loadtxt(predictions_file)
    labels = np.loadtxt(labels_file, dtype=int)
    
    evaluator = CalibrationEvaluator()
    metrics = evaluator.evaluate(predictions, labels)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot reliability diagram
    evaluator.plot_reliability_diagram(
        predictions.tolist(),
        labels.tolist(),
        save_path=str(output_path / "reliability_diagram.png")
    )
    
    # Plot confidence by correctness
    evaluator.plot_confidence_by_correctness(
        predictions.tolist(),
        labels.tolist(),
        save_path=str(output_path / "confidence_by_correctness.png")
    )
    
    # Save metrics
    import json
    with open(output_path / "calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Calibration results saved to {output_dir}")
    
    return metrics
