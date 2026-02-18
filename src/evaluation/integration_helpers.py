"""
Helper functions to integrate selective prediction and conformal prediction
into Smart Notes verification reports.

This module provides convenience functions to:
1. Compute selective prediction metrics from verification results
2. Compute conformal prediction thresholds from calibration data
3. Format results for inclusion in reports
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging

from src.evaluation.selective_prediction import (
    selective_prediction_analysis,
    SelectivePredictionResult
)
from src.evaluation.conformal import (
    conformal_prediction_calibration,
    ConformalResult
)

logger = logging.getLogger(__name__)


def compute_selective_prediction_metrics(
    confidence_scores: List[float],
    predictions: List[int],
    targets: List[int],
    target_risk: float = 0.05,
    num_thresholds: int = 100
) -> Dict[str, Any]:
    """
    Compute selective prediction metrics from verification results.
    
    Args:
        confidence_scores: Confidence scores for each prediction
        predictions: Predicted labels (0, 1, 2 for ENTAIL, NEUTRAL, CONTRADICT)
        targets: Ground truth labels
        target_risk: Target maximum error rate (default: 0.05 = 5%)
        num_thresholds: Number of threshold points to evaluate
    
    Returns:
        Dictionary with keys:
        - optimal_threshold: Recommended confidence threshold
        - achieved_coverage: Coverage at optimal threshold
        - achieved_risk: Risk (error rate) at optimal threshold
        - target_risk: Target risk parameter
        - auc_rc: Area under risk-coverage curve
    
    Example:
        >>> scores = [0.9, 0.8, 0.7, 0.6]
        >>> preds = [1, 1, 0, 0]
        >>> targets = [1, 0, 0, 1]  # 50% accuracy
        >>> metrics = compute_selective_prediction_metrics(scores, preds, targets, target_risk=0.1)
        >>> print(f"Threshold: {metrics['optimal_threshold']:.2f}")
    """
    if len(confidence_scores) == 0:
        logger.warning("Empty confidence scores, returning default metrics")
        return {
            'optimal_threshold': 0.5,
            'achieved_coverage': 0.0,
            'achieved_risk': 0.0,
            'target_risk': target_risk,
            'auc_rc': 0.0
        }
    
    # Convert to numpy arrays
    scores = np.array(confidence_scores)
    preds = np.array(predictions)
    tgts = np.array(targets)
    
    # Run selective prediction analysis
    result = selective_prediction_analysis(
        scores, preds, tgts,
        target_risk=target_risk,
        num_thresholds=num_thresholds
    )
    
    # Format for reporting
    return {
        'optimal_threshold': result.optimal_threshold,
        'achieved_coverage': result.achieved_coverage,
        'achieved_risk': result.achieved_risk,
        'target_risk': result.target_risk,
        'auc_rc': result.auc_rc
    }


def compute_conformal_prediction_metrics(
    calibration_scores: List[float],
    calibration_predictions: List[int],
    calibration_targets: List[int],
    alpha: float = 0.1,
    method: str = "simple"
) -> Dict[str, Any]:
    """
    Compute conformal prediction threshold from calibration set.
    
    Args:
        calibration_scores: Confidence scores on calibration set
        calibration_predictions: Predicted labels on calibration set
        calibration_targets: True labels on calibration set
        alpha: Significance level (1 - alpha = confidence level)
        method: "simple" or "full" conformal method
    
    Returns:
        Dictionary with keys:
        - threshold: Calibrated confidence threshold
        - alpha: Significance level
        - coverage_guarantee: Theoretical coverage (1 - alpha)
        - empirical_coverage: Observed coverage on calibration set
        - calibration_size: Number of calibration examples
    
    Example:
        >>> cal_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        >>> cal_preds = np.array([1, 1, 0, 0, 0])
        >>> cal_targets = np.array([1, 0, 0, 1, 0])
        >>> metrics = compute_conformal_prediction_metrics(
        ...     cal_scores, cal_preds, cal_targets, alpha=0.1
        ... )
        >>> print(f"Threshold: {metrics['threshold']:.2f}")
    """
    if len(calibration_scores) == 0:
        logger.warning("Empty calibration set, returning default metrics")
        return {
            'threshold': 0.5,
            'alpha': alpha,
            'coverage_guarantee': 1 - alpha,
            'empirical_coverage': 1.0,
            'calibration_size': 0
        }
    
    # Convert to numpy arrays
    cal_scores = np.array(calibration_scores)
    cal_preds = np.array(calibration_predictions)
    cal_tgts = np.array(calibration_targets)
    
    # Run conformal calibration
    result = conformal_prediction_calibration(
        cal_scores, cal_preds, cal_tgts,
        alpha=alpha,
        method=method
    )
    
    # Format for reporting
    return {
        'threshold': result.threshold,
        'alpha': result.alpha,
        'coverage_guarantee': result.coverage_guarantee,
        'empirical_coverage': result.empirical_coverage,
        'calibration_size': result.calibration_size
    }


def combine_metrics(
    scores: List[float],
    predictions: List[int],
    targets: List[int],
    target_risk: float = 0.05,
    alpha: float = 0.1,
    calibration_split: float = 0.5
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute both selective and conformal metrics with automatic train/cal split.
    
    Args:
        scores: All confidence scores
        predictions: All predicted labels
        targets: All ground truth labels
        target_risk: Target risk for selective prediction
        alpha: Significance level for conformal prediction
        calibration_split: Fraction to use for calibration (default: 0.5)
    
    Returns:
        Tuple of (selective_metrics, conformal_metrics)
    
    Example:
        >>> scores = list(range(100))
        >>> preds = [i % 2 for i in range(100)]
        >>> targets = [(i+1) % 2 for i in range(100)]
        >>> sp_metrics, cp_metrics = combine_metrics(
        ...     scores, preds, targets, target_risk=0.1, alpha=0.1
        ... )
    """
    if len(scores) == 0:
        return ({}, {})
    
    # Split into calibration and rest
    n = len(scores)
    n_cal = int(n * calibration_split)
    
    if n_cal < 10:
        logger.warning(
            f"Calibration set too small (n={n_cal}), using all data for both metrics"
        )
        cal_scores = scores
        cal_preds = predictions
        cal_targets = targets
        test_scores = scores
        test_preds = predictions
        test_targets = targets
    else:
        # Random split (should use fixed seed in production)
        indices = np.random.permutation(n)
        cal_indices = indices[:n_cal]
        test_indices = indices[n_cal:]
        
        scores_arr = np.array(scores)
        preds_arr = np.array(predictions)
        targets_arr = np.array(targets)
        
        cal_scores = scores_arr[cal_indices].tolist()
        cal_preds = preds_arr[cal_indices].tolist()
        cal_targets = targets_arr[cal_indices].tolist()
        
        test_scores = scores_arr[test_indices].tolist()
        test_preds = preds_arr[test_indices].tolist()
        test_targets = targets_arr[test_indices].tolist()
    
    # Compute selective prediction on test set
    selective_metrics = compute_selective_prediction_metrics(
        test_scores, test_preds, test_targets, target_risk=target_risk
    )
    
    # Compute conformal prediction on calibration set
    conformal_metrics = compute_conformal_prediction_metrics(
        cal_scores, cal_preds, cal_targets, alpha=alpha
    )
    
    logger.info(
        f"Computed metrics: selective_threshold={selective_metrics['optimal_threshold']:.3f}, "
        f"conformal_threshold={conformal_metrics['threshold']:.3f}"
    )
    
    return selective_metrics, conformal_metrics


def add_confidence_guarantees_to_verification_summary(
    verification_summary: Dict[str, Any],
    scores: List[float],
    predictions: List[int],
    targets: List[int],
    target_risk: float = 0.05,
    alpha: float = 0.1
) -> Dict[str, Any]:
    """
    Add selective and conformal prediction metrics to existing verification summary.
    
    Args:
        verification_summary: Existing verification summary dict
        scores: Confidence scores
        predictions: Predicted labels
        targets: Ground truth labels
        target_risk: Target risk for selective prediction
        alpha: Significance level for conformal prediction
    
    Returns:
        Updated verification summary with added fields:
        - selective_prediction: Dict with selective prediction metrics
        - conformal_prediction: Dict with conformal prediction metrics
    
    Example:
        >>> summary = {
        ...     'total_claims': 100,
        ...     'verified_count': 80,
        ...     'avg_confidence': 0.85
        ... }
        >>> enhanced = add_confidence_guarantees_to_verification_summary(
        ...     summary, scores, preds, targets
        ... )
        >>> print(enhanced['selective_prediction']['optimal_threshold'])
    """
    # Compute metrics
    selective_metrics, conformal_metrics = combine_metrics(
        scores, predictions, targets,
        target_risk=target_risk,
        alpha=alpha
    )
    
    # Add to summary
    updated_summary = verification_summary.copy()
    updated_summary['selective_prediction'] = selective_metrics
    updated_summary['conformal_prediction'] = conformal_metrics
    
    return updated_summary


def format_threshold_recommendation(
    selective_metrics: Dict[str, Any],
    conformal_metrics: Dict[str, Any]
) -> str:
    """
    Format a human-readable recommendation for threshold usage.
    
    Args:
        selective_metrics: Selective prediction metrics
        conformal_metrics: Conformal prediction metrics
    
    Returns:
        Formatted string with recommendations
    """
    sp_threshold = selective_metrics.get('optimal_threshold', 0.5)
    cp_threshold = conformal_metrics.get('threshold', 0.5)
    combined_threshold = max(sp_threshold, cp_threshold)
    
    coverage = selective_metrics.get('achieved_coverage', 0.0)
    risk = selective_metrics.get('achieved_risk', 0.0)
    confidence_level = conformal_metrics.get('coverage_guarantee', 0.9)
    
    recommendation = f"""
CONFIDENCE THRESHOLD RECOMMENDATION
{'='*60}

Recommended Threshold: {combined_threshold:.3f}

At this threshold:
  • {coverage:.1%} of predictions will be accepted
  • Expected error rate: {risk:.1%}
  • {confidence_level:.1%} confidence guarantee

Usage Guidelines:
  ✅ ACCEPT predictions with confidence ≥ {combined_threshold:.3f}
  ⚠️ REVIEW predictions with {min(sp_threshold, cp_threshold):.3f} ≤ confidence < {combined_threshold:.3f}
  ❌ REJECT predictions with confidence < {min(sp_threshold, cp_threshold):.3f}

Technical Details:
  • Selective Prediction threshold: {sp_threshold:.3f}
  • Conformal Prediction threshold: {cp_threshold:.3f}
  • Combined (conservative) threshold: {combined_threshold:.3f}
"""
    
    return recommendation
