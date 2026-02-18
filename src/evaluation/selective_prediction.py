"""
Selective Prediction for Claim Verification

Implements selective prediction (prediction with rejection option) to allow
the system to abstain on uncertain predictions while providing risk-coverage
tradeoffs.

Key Concepts:
- Risk: Error rate on accepted predictions
- Coverage: Fraction of predictions accepted
- Risk-Coverage Curve: Tradeoff between risk and coverage at different thresholds

References:
- Geifman & El-Yaniv (2017): "Selective Prediction for Deep Neural Networks"
- Wiener & El-Yaniv (2011): "Risk-Coverage Curves"
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskCoveragePoint:
    """A single point on the risk-coverage curve."""
    threshold: float
    coverage: float  # Fraction of examples accepted (0-1)
    risk: float  # Error rate on accepted examples (0-1)
    num_accepted: int
    num_correct: int
    num_errors: int


@dataclass
class SelectivePredictionResult:
    """Result of selective prediction analysis."""
    risk_coverage_curve: List[RiskCoveragePoint]
    optimal_threshold: float
    target_risk: float
    achieved_risk: float
    achieved_coverage: float
    auc_rc: float  # Area under risk-coverage curve


def compute_risk_coverage_curve(
    scores: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_thresholds: int = 100
) -> List[RiskCoveragePoint]:
    """
    Compute risk-coverage curve by varying confidence threshold.
    
    Algorithm:
    1. Sort examples by confidence score (descending)
    2. For each threshold, compute:
       - Coverage: fraction of examples with score >= threshold
       - Risk: error rate on accepted examples
    
    Args:
        scores: Confidence scores (higher = more confident), shape (n,)
        predictions: Predicted labels, shape (n,)
        targets: Ground truth labels, shape (n,)
        num_thresholds: Number of threshold points to evaluate
    
    Returns:
        List of RiskCoveragePoint objects sorted by decreasing coverage
    
    Example:
        >>> scores = np.array([0.9, 0.8, 0.6, 0.4])
        >>> predictions = np.array([1, 1, 0, 0])
        >>> targets = np.array([1, 0, 0, 1])  # 2nd and 4th are errors
        >>> curve = compute_risk_coverage_curve(scores, predictions, targets)
        >>> # At threshold 0.85: accepts only 1st, risk=0.0, coverage=0.25
        >>> # At threshold 0.7: accepts 1st+2nd, risk=0.5, coverage=0.5
        >>> # At threshold 0.5: accepts 1st+2nd+3rd, risk=0.33, coverage=0.75
    """
    if len(scores) != len(predictions) or len(scores) != len(targets):
        raise ValueError("scores, predictions, and targets must have same length")
    
    if len(scores) == 0:
        return []
    
    # Create array of thresholds from min to max score
    min_score = np.min(scores)
    max_score = np.max(scores)
    thresholds = np.linspace(min_score, max_score, num_thresholds)
    
    # Add boundary thresholds
    thresholds = np.concatenate([
        [min_score - 0.01],  # Accept all
        thresholds,
        [max_score + 0.01]  # Accept none
    ])
    thresholds = np.unique(thresholds)
    
    curve = []
    n = len(scores)
    
    for threshold in thresholds:
        # Accept examples with score >= threshold
        accepted_mask = scores >= threshold
        num_accepted = np.sum(accepted_mask)
        
        if num_accepted == 0:
            # No examples accepted
            coverage = 0.0
            risk = 0.0  # Undefined, but set to 0
            num_correct = 0
            num_errors = 0
        else:
            # Compute risk on accepted examples
            accepted_preds = predictions[accepted_mask]
            accepted_targets = targets[accepted_mask]
            
            num_correct = np.sum(accepted_preds == accepted_targets)
            num_errors = num_accepted - num_correct
            
            coverage = num_accepted / n
            risk = num_errors / num_accepted if num_accepted > 0 else 0.0
        
        point = RiskCoveragePoint(
            threshold=float(threshold),
            coverage=float(coverage),
            risk=float(risk),
            num_accepted=int(num_accepted),
            num_correct=int(num_correct),
            num_errors=int(num_errors)
        )
        curve.append(point)
    
    # Sort by decreasing coverage (increasing threshold)
    curve.sort(key=lambda p: p.coverage, reverse=True)
    
    return curve


def find_threshold_for_target_risk(
    risk_coverage_curve: List[RiskCoveragePoint],
    target_risk: float,
    prefer_higher_coverage: bool = True
) -> Tuple[float, float, float]:
    """
    Find threshold that achieves target risk (or better).
    
    Args:
        risk_coverage_curve: Computed risk-coverage curve
        target_risk: Maximum acceptable risk (error rate)
        prefer_higher_coverage: If multiple thresholds achieve target,
                               choose one with higher coverage
    
    Returns:
        Tuple of (threshold, achieved_risk, achieved_coverage)
    
    Example:
        >>> curve = [...]  # Risk-coverage curve
        >>> threshold, risk, cov = find_threshold_for_target_risk(curve, target_risk=0.1)
        >>> print(f"At threshold {threshold:.2f}: risk={risk:.2f}, coverage={cov:.2f}")
    """
    if not risk_coverage_curve:
        raise ValueError("Empty risk-coverage curve")
    
    # Find points where risk <= target_risk
    valid_points = [p for p in risk_coverage_curve if p.risk <= target_risk]
    
    if not valid_points:
        # No threshold achieves target risk, return most conservative
        logger.warning(
            f"Cannot achieve target risk {target_risk:.3f}. "
            f"Minimum achievable risk: {min(p.risk for p in risk_coverage_curve):.3f}"
        )
        # Return point with lowest risk
        best_point = min(risk_coverage_curve, key=lambda p: p.risk)
        return best_point.threshold, best_point.risk, best_point.coverage
    
    # Choose point with highest coverage among valid points
    if prefer_higher_coverage:
        best_point = max(valid_points, key=lambda p: p.coverage)
    else:
        best_point = min(valid_points, key=lambda p: p.risk)
    
    return best_point.threshold, best_point.risk, best_point.coverage


def compute_auc_risk_coverage(
    risk_coverage_curve: List[RiskCoveragePoint]
) -> float:
    """
    Compute area under risk-coverage curve (AUC-RC).
    
    Lower AUC-RC is better (less risk for same coverage).
    
    Args:
        risk_coverage_curve: Risk-coverage curve points
    
    Returns:
        Area under curve (trapezoid integration)
    """
    if not risk_coverage_curve or len(risk_coverage_curve) < 2:
        return 0.0
    
    # Sort by coverage (increasing)
    sorted_curve = sorted(risk_coverage_curve, key=lambda p: p.coverage)
    
    # Trapezoid rule integration
    auc = 0.0
    for i in range(len(sorted_curve) - 1):
        p1 = sorted_curve[i]
        p2 = sorted_curve[i + 1]
        
        width = p2.coverage - p1.coverage
        avg_height = (p1.risk + p2.risk) / 2
        auc += width * avg_height
    
    return auc


def selective_prediction_analysis(
    scores: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_risk: float = 0.05,
    num_thresholds: int = 100
) -> SelectivePredictionResult:
    """
    Perform complete selective prediction analysis.
    
    Args:
        scores: Confidence scores, shape (n,)
        predictions: Predicted labels, shape (n,)
        targets: Ground truth labels, shape (n,)
        target_risk: Target maximum risk (default: 0.05 = 5% error)
        num_thresholds: Number of thresholds to evaluate
    
    Returns:
        SelectivePredictionResult with full analysis
    
    Example:
        >>> result = selective_prediction_analysis(scores, preds, targets, target_risk=0.1)
        >>> print(f"Optimal threshold: {result.optimal_threshold:.2f}")
        >>> print(f"Coverage: {result.achieved_coverage:.1%}")
        >>> print(f"Risk: {result.achieved_risk:.1%}")
    """
    # Compute risk-coverage curve
    curve = compute_risk_coverage_curve(
        scores, predictions, targets, num_thresholds
    )
    
    # Find optimal threshold for target risk
    threshold, risk, coverage = find_threshold_for_target_risk(
        curve, target_risk
    )
    
    # Compute AUC
    auc = compute_auc_risk_coverage(curve)
    
    result = SelectivePredictionResult(
        risk_coverage_curve=curve,
        optimal_threshold=threshold,
        target_risk=target_risk,
        achieved_risk=risk,
        achieved_coverage=coverage,
        auc_rc=auc
    )
    
    logger.info(
        f"Selective prediction: threshold={threshold:.3f}, "
        f"coverage={coverage:.1%}, risk={risk:.3f} (target: {target_risk:.3f})"
    )
    
    return result


def apply_selective_prediction(
    scores: np.ndarray,
    predictions: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply selective prediction threshold to reject low-confidence predictions.
    
    Args:
        scores: Confidence scores, shape (n,)
        predictions: Predicted labels, shape (n,)
        threshold: Confidence threshold
    
    Returns:
        Tuple of (accepted_indices, rejected_indices)
    
    Example:
        >>> scores = np.array([0.9, 0.7, 0.4])
        >>> predictions = np.array([1, 1, 0])
        >>> accepted, rejected = apply_selective_prediction(scores, predictions, threshold=0.6)
        >>> print(f"Accepted: {accepted}, Rejected: {rejected}")
        Accepted: [0, 1], Rejected: [2]
    """
    accepted_mask = scores >= threshold
    accepted_indices = np.where(accepted_mask)[0]
    rejected_indices = np.where(~accepted_mask)[0]
    
    return accepted_indices, rejected_indices


def get_risk_for_coverage(
    risk_coverage_curve: List[RiskCoveragePoint],
    target_coverage: float
) -> float:
    """
    Get minimum risk achievable at given coverage level.
    
    Args:
        risk_coverage_curve: Risk-coverage curve
        target_coverage: Desired coverage level (0-1)
    
    Returns:
        Minimum risk at target coverage
    """
    if not risk_coverage_curve:
        return 1.0
    
    # Find point with coverage closest to target (but >= target)
    valid_points = [p for p in risk_coverage_curve if p.coverage >= target_coverage]
    
    if not valid_points:
        # No coverage achieves target, return highest coverage point
        return max(risk_coverage_curve, key=lambda p: p.coverage).risk
    
    # Return point with coverage closest to target
    best_point = min(valid_points, key=lambda p: abs(p.coverage - target_coverage))
    return best_point.risk


def get_coverage_for_risk(
    risk_coverage_curve: List[RiskCoveragePoint],
    target_risk: float
) -> float:
    """
    Get maximum coverage achievable at given risk level.
    
    Args:
        risk_coverage_curve: Risk-coverage curve
        target_risk: Maximum acceptable risk (0-1)
    
    Returns:
        Maximum coverage at target risk
    """
    if not risk_coverage_curve:
        return 0.0
    
    # Find points with risk <= target
    valid_points = [p for p in risk_coverage_curve if p.risk <= target_risk]
    
    if not valid_points:
        # No point achieves target risk
        return 0.0
    
    # Return highest coverage
    return max(p.coverage for p in valid_points)


def format_selective_prediction_summary(
    result: SelectivePredictionResult
) -> str:
    """
    Format selective prediction results as human-readable summary.
    
    Args:
        result: SelectivePredictionResult
    
    Returns:
        Formatted string summary
    """
    summary = [
        "=" * 60,
        "SELECTIVE PREDICTION ANALYSIS",
        "=" * 60,
        f"Target Risk: {result.target_risk:.1%} (max acceptable error rate)",
        f"",
        f"Recommended Threshold: {result.optimal_threshold:.3f}",
        f"  → Coverage: {result.achieved_coverage:.1%} of examples accepted",
        f"  → Risk: {result.achieved_risk:.1%} error rate on accepted",
        f"  → Rejected: {(1-result.achieved_coverage):.1%} of examples",
        f"",
        f"AUC-RC: {result.auc_rc:.4f} (lower is better)",
        "=" * 60
    ]
    
    return "\n".join(summary)


def plot_risk_coverage_curve(
    result: SelectivePredictionResult,
    save_path: Optional[str] = None
) -> Any:
    """
    Plot risk-coverage curve.
    
    Args:
        result: SelectivePredictionResult
        save_path: Optional path to save plot
    
    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, cannot plot")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    curve = result.risk_coverage_curve
    coverages = [p.coverage for p in curve]
    risks = [p.risk for p in curve]
    
    # Plot curve
    ax.plot(coverages, risks, 'b-', linewidth=2, label='Risk-Coverage Curve')
    
    # Mark optimal point
    ax.plot(
        result.achieved_coverage,
        result.achieved_risk,
        'ro',
        markersize=10,
        label=f'Optimal (threshold={result.optimal_threshold:.2f})'
    )
    
    # Mark target risk line
    ax.axhline(
        y=result.target_risk,
        color='r',
        linestyle='--',
        alpha=0.5,
        label=f'Target Risk ({result.target_risk:.1%})'
    )
    
    ax.set_xlabel('Coverage (fraction of examples accepted)', fontsize=12)
    ax.set_ylabel('Risk (error rate on accepted)', fontsize=12)
    ax.set_title('Risk-Coverage Tradeoff', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(risks) * 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved risk-coverage plot to {save_path}")
    
    return fig
