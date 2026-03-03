"""
Bootstrap Confidence Intervals for Evaluation Metrics.

Computes 95% confidence intervals using bootstrap resampling
for the following metrics:
- Accuracy
- Macro-F1
- ECE (Expected Calibration Error)
- AUC-AC (Area Under Accuracy-Coverage curve)

References:
- Efron & Tibshirani (1993): An Introduction to the Bootstrap
- DiCiccio & Efron (1996): Bootstrap Confidence Intervals
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass, asdict
import logging
from sklearn.metrics import f1_score
from src.evaluation.calibration import CalibrationEvaluator
from src.evaluation.selective_prediction import compute_auc_accuracy_coverage

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""
    metric_name: str
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_bootstrap: int
    
    def __str__(self) -> str:
        return f"{self.metric_name}: {self.point_estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


@dataclass
class BootstrapCIReport:
    """Complete bootstrap CI report."""
    accuracy: ConfidenceInterval
    macro_f1: ConfidenceInterval
    ece: ConfidenceInterval
    auc_ac: ConfidenceInterval
    n_samples: int
    confidence_level: float
    n_bootstrap: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "accuracy": asdict(self.accuracy),
            "macro_f1": asdict(self.macro_f1),
            "ece": asdict(self.ece),
            "auc_ac": asdict(self.auc_ac),
            "n_samples": self.n_samples,
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap
        }
    
    def summary_table(self) -> str:
        """Generate markdown table."""
        lines = [
            "| Metric | Value | 95% CI |",
            "|--------|-------|--------|",
            f"| Accuracy | {self.accuracy.point_estimate:.4f} | [{self.accuracy.lower:.4f}, {self.accuracy.upper:.4f}] |",
            f"| Macro-F1 | {self.macro_f1.point_estimate:.4f} | [{self.macro_f1.lower:.4f}, {self.macro_f1.upper:.4f}] |",
            f"| ECE (15 bins) | {self.ece.point_estimate:.4f} | [{self.ece.lower:.4f}, {self.ece.upper:.4f}] |",
            f"| AUC-AC | {self.auc_ac.point_estimate:.4f} | [{self.auc_ac.lower:.4f}, {self.auc_ac.upper:.4f}] |"
        ]
        return "\n".join(lines)


def bootstrap_ci(
    data: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Uses percentile bootstrap method (BCa not implemented for simplicity).
    
    Args:
        data: Input data (will be resampled with replacement)
        metric_fn: Function that computes metric from data
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(random_seed)
    n = len(data)
    
    # Compute point estimate
    point_estimate = metric_fn(data)
    
    # Bootstrap resampling
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        resampled_data = data[indices]
        
        try:
            estimate = metric_fn(resampled_data)
            bootstrap_estimates.append(estimate)
        except Exception as e:
            logger.warning(f"Bootstrap iteration failed: {e}")
            continue
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    # Compute percentile CI
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    lower = np.percentile(bootstrap_estimates, lower_percentile)
    upper = np.percentile(bootstrap_estimates, upper_percentile)
    
    return point_estimate, lower, upper


def compute_accuracy(indices_and_data: np.ndarray) -> float:
    """Compute accuracy from paired predictions and labels."""
    predictions, labels = indices_and_data[:, 0], indices_and_data[:, 1]
    return np.mean(predictions == labels)


def compute_macro_f1(indices_and_data: np.ndarray) -> float:
    """Compute macro-F1 from paired predictions and labels."""
    predictions, labels = indices_and_data[:, 0], indices_and_data[:, 1]
    
    # Convert to integer labels
    predictions = predictions.astype(int)
    labels = labels.astype(int)
    
    # Get unique labels present in this bootstrap sample
    unique_labels = np.unique(np.concatenate([predictions, labels]))
    
    if len(unique_labels) == 0:
        return 0.0
    
    return f1_score(labels, predictions, labels=unique_labels, average='macro', zero_division=0)


def compute_ece_from_data(conf_label_correct: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute ECE from confidence scores and correctness labels.
    
    Args:
        conf_label_correct: Array of shape (n, 2) where:
            - col 0: confidence score
            - col 1: correctness (1 if prediction correct, 0 otherwise)
        n_bins: Number of bins for ECE calculation
    
    Returns:
        ECE value
    """
    confidences = conf_label_correct[:, 0]
    correctness = conf_label_correct[:, 1]
    
    evaluator = CalibrationEvaluator(n_bins=n_bins)
    ece = evaluator.expected_calibration_error(confidences, correctness)
    return ece


def compute_auc_ac_from_data(conf_pred_label: np.ndarray) -> float:
    """
    Compute AUC-AC from confidence scores, predictions, and labels.
    
    Args:
        conf_pred_label: Array of shape (n, 3) where:
            - col 0: confidence score
            - col 1: predicted label
            - col 2: true label
    
    Returns:
        AUC-AC value
    """
    confidences = conf_pred_label[:, 0]
    predictions = conf_pred_label[:, 1].astype(int)
    labels = conf_pred_label[:, 2].astype(int)
    
    # Compute correctness
    correctness = (predictions == labels).astype(int)
    
    try:
        auc_ac = compute_auc_accuracy_coverage(confidences, correctness)
        return auc_ac
    except Exception as e:
        logger.warning(f"Failed to compute AUC-AC: {e}")
        return 0.0


def compute_bootstrap_cis(
    predictions: List[int],
    labels: List[int],
    confidences: List[float],
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    n_bins_ece: int = 15,
    random_seed: int = 42
) -> BootstrapCIReport:
    """
    Compute bootstrap confidence intervals for all key metrics.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        confidences: Confidence scores
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 95%)
        n_bins_ece: Number of bins for ECE calculation
        random_seed: Random seed for reproducibility
    
    Returns:
        BootstrapCIReport with all CIs
    """
    logger.info(f"Computing bootstrap CIs with {n_bootstrap} samples...")
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    confidences = np.array(confidences)
    n_samples = len(predictions)
    
    # Prepare data arrays
    # For accuracy and F1: (prediction, label)
    pred_label_data = np.column_stack([predictions, labels])
    
    # For ECE: (confidence, correctness)
    correctness = (predictions == labels).astype(int)
    conf_correct_data = np.column_stack([confidences, correctness])
    
    # For AUC-AC: (confidence, prediction, label)
    conf_pred_label_data = np.column_stack([confidences, predictions, labels])
    
    # Compute CIs
    logger.info("Computing accuracy CI...")
    acc_point, acc_lower, acc_upper = bootstrap_ci(
        pred_label_data,
        compute_accuracy,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed
    )
    
    logger.info("Computing macro-F1 CI...")
    f1_point, f1_lower, f1_upper = bootstrap_ci(
        pred_label_data,
        compute_macro_f1,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed + 1
    )
    
    logger.info("Computing ECE CI...")
    
    def ece_metric(data):
        return compute_ece_from_data(data, n_bins=n_bins_ece)
    
    ece_point, ece_lower, ece_upper = bootstrap_ci(
        conf_correct_data,
        ece_metric,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed + 2
    )
    
    logger.info("Computing AUC-AC CI...")
    auc_point, auc_lower, auc_upper = bootstrap_ci(
        conf_pred_label_data,
        compute_auc_ac_from_data,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed + 3
    )
    
    # Create report
    report = BootstrapCIReport(
        accuracy=ConfidenceInterval(
            metric_name="Accuracy",
            point_estimate=acc_point,
            lower=acc_lower,
            upper=acc_upper,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap
        ),
        macro_f1=ConfidenceInterval(
            metric_name="Macro-F1",
            point_estimate=f1_point,
            lower=f1_lower,
            upper=f1_upper,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap
        ),
        ece=ConfidenceInterval(
            metric_name="ECE",
            point_estimate=ece_point,
            lower=ece_lower,
            upper=ece_upper,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap
        ),
        auc_ac=ConfidenceInterval(
            metric_name="AUC-AC",
            point_estimate=auc_point,
            lower=auc_lower,
            upper=auc_upper,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap
        ),
        n_samples=n_samples,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap
    )
    
    logger.info("Bootstrap CI computation complete:")
    logger.info(f"  {report.accuracy}")
    logger.info(f"  {report.macro_f1}")
    logger.info(f"  {report.ece}")
    logger.info(f"  {report.auc_ac}")
    
    return report


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n = 100
    predictions = np.random.randint(0, 3, n)
    labels = np.random.randint(0, 3, n)
    confidences = np.random.uniform(0.3, 1.0, n)
    
    report = compute_bootstrap_cis(
        predictions=predictions.tolist(),
        labels=labels.tolist(),
        confidences=confidences.tolist(),
        n_bootstrap=1000
    )
    
    print("\n" + report.summary_table())
