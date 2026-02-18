"""
Conformal Prediction for Claim Verification

Implements conformal prediction to provide distribution-free confidence guarantees
on prediction sets or error rates.

Key Concepts:
- Conformal prediction provides finite-sample guarantees without distributional assumptions
- Uses calibration set to compute nonconformity scores
- Provides prediction sets with coverage guarantee: P(y ∈ C(x)) ≥ 1 - α

References:
- Vovk et al. (2005): "Algorithmic Learning in a Random World"
- Angelopoulos & Bates (2021): "A Gentle Introduction to Conformal Prediction"
- Romano et al. (2020): "Classification with Valid and Adaptive Coverage"
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConformalResult:
    """Result of conformal prediction calibration."""
    threshold: float  # Calibrated threshold
    alpha: float  # Significance level (1 - alpha = confidence)
    coverage_guarantee: float  # Theoretical coverage (1 - alpha)
    empirical_coverage: float  # Observed coverage on calibration set
    calibration_size: int  # Number of calibration examples
    num_rejected: int  # Number of examples below threshold


def compute_conformal_threshold(
    scores: np.ndarray,
    is_correct: np.ndarray,
    alpha: float = 0.1
) -> float:
    """
    Compute conformal prediction threshold for error control.
    
    Uses the split conformal approach to find a threshold such that
    predictions with score >= threshold have error rate <= alpha
    with probability at least 1 - alpha.
    
    Algorithm (Simple Conformal):
    1. Compute nonconformity scores on calibration set
    2. Find (1-alpha) quantile of scores for incorrect predictions
    3. Set threshold to this quantile
    
    Args:
        scores: Confidence scores on calibration set, shape (n,)
        is_correct: Whether prediction is correct (1) or not (0), shape (n,)
        alpha: Significance level (default: 0.1 for 90% confidence)
    
    Returns:
        Calibrated threshold for selective prediction
    
    Example:
        >>> scores_cal = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        >>> is_correct_cal = np.array([1, 1, 0, 0, 0])
        >>> threshold = compute_conformal_threshold(scores_cal, is_correct_cal, alpha=0.1)
        >>> # Predictions with score >= threshold will have bounded error
    """
    if len(scores) != len(is_correct):
        raise ValueError("scores and is_correct must have same length")
    
    if len(scores) == 0:
        raise ValueError("Empty calibration set")
    
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    
    n = len(scores)
    
    # Nonconformity scores: use negative score for incorrect predictions
    # (higher nonconformity = more surprising/unusual)
    # For correct predictions, nonconformity is low (high score)
    # For incorrect predictions, nonconformity is high (low score inverted to high)
    nonconformity_scores = np.where(
        is_correct,
        -scores,  # Correct: low nonconformity (high confidence)
        1 - scores  # Incorrect: high nonconformity (low confidence)
    )
    
    # Compute (1-alpha)(n+1) quantile (adjusted for finite sample)
    # This ensures coverage guarantee holds
    level = np.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)  # Cap at 1.0
    
    threshold_nonconformity = np.quantile(nonconformity_scores, level)
    
    # Convert back to score threshold
    # We want: for accepted predictions, nonconformity <= threshold_nonconformity
    # For predictions with score >= threshold, we have:
    #   - If correct: -score <= threshold_nonconformity → score >= -threshold_nonconformity
    #   - If incorrect: 1-score <= threshold_nonconformity → score >= 1-threshold_nonconformity
    # To be conservative, use the tighter bound
    threshold_score = 1 - threshold_nonconformity
    
    # Ensure threshold is in valid range [0, 1]
    threshold_score = np.clip(threshold_score, 0.0, 1.0)
    
    logger.debug(
        f"Conformal threshold: {threshold_score:.3f} "
        f"(alpha={alpha}, n={n}, level={level:.3f})"
    )
    
    return float(threshold_score)


def compute_conformal_threshold_simple(
    scores: np.ndarray,
    is_correct: np.ndarray,
    alpha: float = 0.1
) -> float:
    """
    Simplified conformal threshold using score quantile on errors.
    
    This is a simpler, more intuitive version:
    - Find scores of incorrect predictions
    - Set threshold at (1-alpha) quantile of these error scores
    - Guarantees: if we reject predictions below threshold,
      error rate will be controlled
    
    Args:
        scores: Confidence scores, shape (n,)
        is_correct: Correctness indicators, shape (n,)
        alpha: Significance level
    
    Returns:
        Calibrated threshold
    """
    if len(scores) == 0:
        return 0.5
    
    # Get scores of errors
    error_scores = scores[is_correct == 0]
    
    if len(error_scores) == 0:
        # No errors in calibration set
        logger.warning("No errors in calibration set, using minimum score")
        return float(np.min(scores))
    
    # Compute quantile with finite-sample correction
    n_errors = len(error_scores)
    level = (1 - alpha) * (n_errors + 1) / n_errors
    level = min(level, 1.0)
    
    threshold = np.quantile(error_scores, level)
    
    return float(threshold)


def validate_conformal_coverage(
    test_scores: np.ndarray,
    test_is_correct: np.ndarray,
    threshold: float
) -> Tuple[float, float, int, int]:
    """
    Validate conformal prediction on test set.
    
    Args:
        test_scores: Confidence scores on test set
        test_is_correct: Correctness on test set
        threshold: Calibrated threshold
    
    Returns:
        Tuple of (error_rate, coverage, num_accepted, num_errors)
    """
    if len(test_scores) == 0:
        return 0.0, 0.0, 0, 0
    
    # Accept predictions above threshold
    accepted_mask = test_scores >= threshold
    num_accepted = np.sum(accepted_mask)
    
    if num_accepted == 0:
        return 0.0, 0.0, 0, 0
    
    # Compute error rate on accepted
    accepted_correct = test_is_correct[accepted_mask]
    num_errors = np.sum(accepted_correct == 0)
    error_rate = num_errors / num_accepted
    
    coverage = num_accepted / len(test_scores)
    
    return float(error_rate), float(coverage), int(num_accepted), int(num_errors)


def conformal_prediction_calibration(
    cal_scores: np.ndarray,
    cal_predictions: np.ndarray,
    cal_targets: np.ndarray,
    alpha: float = 0.1,
    method: str = "simple"
) -> ConformalResult:
    """
    Perform conformal prediction calibration.
    
    Args:
        cal_scores: Confidence scores on calibration set
        cal_predictions: Predicted labels on calibration set
        cal_targets: True labels on calibration set
        alpha: Significance level (1 - alpha = confidence)
        method: "simple" or "full" conformal method
    
    Returns:
        ConformalResult with calibrated threshold and statistics
    
    Example:
        >>> result = conformal_prediction_calibration(
        ...     cal_scores, cal_preds, cal_targets, alpha=0.1
        ... )
        >>> print(f"Threshold: {result.threshold:.3f}")
        >>> print(f"Coverage guarantee: {result.coverage_guarantee:.1%}")
    """
    # Compute correctness
    is_correct = (cal_predictions == cal_targets).astype(int)
    
    # Compute threshold
    if method == "simple":
        threshold = compute_conformal_threshold_simple(cal_scores, is_correct, alpha)
    else:
        threshold = compute_conformal_threshold(cal_scores, is_correct, alpha)
    
    # Compute empirical coverage on calibration set
    error_rate, coverage, num_accepted, num_errors = validate_conformal_coverage(
        cal_scores, is_correct, threshold
    )
    
    result = ConformalResult(
        threshold=threshold,
        alpha=alpha,
        coverage_guarantee=1 - alpha,
        empirical_coverage=1 - error_rate if num_accepted > 0 else 1.0,
        calibration_size=len(cal_scores),
        num_rejected=len(cal_scores) - num_accepted
    )
    
    logger.info(
        f"Conformal calibration: threshold={threshold:.3f}, "
        f"coverage_guarantee={result.coverage_guarantee:.1%}, "
        f"empirical_coverage={result.empirical_coverage:.1%}"
    )
    
    return result


def confidence_band_quantile(
    scores: np.ndarray,
    alpha: float = 0.1
) -> Tuple[float, float]:
    """
    Compute confidence band using quantiles.
    
    Returns lower and upper quantiles for (1-alpha) confidence.
    
    Args:
        scores: Score distribution
        alpha: Significance level
    
    Returns:
        Tuple of (lower_quantile, upper_quantile)
    """
    if len(scores) == 0:
        return 0.0, 1.0
    
    lower = np.quantile(scores, alpha / 2)
    upper = np.quantile(scores, 1 - alpha / 2)
    
    return float(lower), float(upper)


def expected_calibration_error(
    scores: np.ndarray,
    is_correct: np.ndarray,
    num_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures calibration quality: how well predicted confidences
    match actual correctness frequencies.
    
    Args:
        scores: Confidence scores
        is_correct: Correctness indicators
        num_bins: Number of bins for binning
    
    Returns:
        ECE score (lower is better, 0 is perfect calibration)
    """
    if len(scores) == 0:
        return 0.0
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    ece = 0.0
    n = len(scores)
    
    for i in range(num_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        # Find examples in this bin
        in_bin = (scores >= lower) & (scores < upper)
        if i == num_bins - 1:  # Last bin includes upper boundary
            in_bin = (scores >= lower) & (scores <= upper)
        
        bin_count = np.sum(in_bin)
        
        if bin_count == 0:
            continue
        
        # Average confidence in bin
        avg_confidence = np.mean(scores[in_bin])
        
        # Average accuracy in bin
        avg_accuracy = np.mean(is_correct[in_bin])
        
        # Weighted difference
        ece += (bin_count / n) * abs(avg_confidence - avg_accuracy)
    
    return float(ece)


def format_conformal_summary(result: ConformalResult) -> str:
    """
    Format conformal prediction results as summary.
    
    Args:
        result: ConformalResult
    
    Returns:
        Formatted string
    """
    summary = [
        "=" * 60,
        "CONFORMAL PREDICTION CALIBRATION",
        "=" * 60,
        f"Confidence Level: {result.coverage_guarantee:.1%} (1 - α where α={result.alpha})",
        f"",
        f"Calibrated Threshold: {result.threshold:.3f}",
        f"",
        f"Guarantee: With {result.coverage_guarantee:.1%} confidence,",
        f"  predictions with score ≥ {result.threshold:.3f} will have",
        f"  error rate ≤ {result.alpha:.1%}",
        f"",
        f"Calibration Statistics:",
        f"  - Calibration Set Size: {result.calibration_size}",
        f"  - Empirical Coverage: {result.empirical_coverage:.1%}",
        f"  - Rejected: {result.num_rejected} examples",
        "=" * 60
    ]
    
    return "\n".join(summary)


def combine_selective_and_conformal(
    scores: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    alpha: float = 0.1,
    target_risk: Optional[float] = None
) -> Dict[str, Any]:
    """
    Combine selective prediction and conformal prediction.
    
    Provides both:
    1. Risk-coverage tradeoff (selective prediction)
    2. Finite-sample guarantees (conformal prediction)
    
    Args:
        scores: Confidence scores
        predictions: Predicted labels
        targets: True labels
        alpha: Significance level for conformal
        target_risk: Optional target risk for selective prediction
    
    Returns:
        Dictionary with both analyses
    """
    from src.evaluation.selective_prediction import selective_prediction_analysis
    
    # Compute correctness
    is_correct = (predictions == targets).astype(int)
    
    # Conformal calibration
    conformal_result = conformal_prediction_calibration(
        scores, predictions, targets, alpha=alpha
    )
    
    # Selective prediction analysis
    if target_risk is None:
        target_risk = alpha  # Use same as conformal
    
    selective_result = selective_prediction_analysis(
        scores, predictions, targets, target_risk=target_risk
    )
    
    return {
        "conformal": conformal_result,
        "selective": selective_result,
        "combined_threshold": max(
            conformal_result.threshold,
            selective_result.optimal_threshold
        )
    }
