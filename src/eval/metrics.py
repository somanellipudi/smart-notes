"""
Unified metric computation module for CalibraTeach.

This module ensures all reported metrics use a single, authoritative definition:
- ECE (Expected Calibration Error): Confidence vs. correctness bins
- AUC-AC (Area Under Accuracy-Coverage Curve): Confidence-based abstention
- Selective prediction metrics: Coverage, accuracy, precision

Single source of truth for all paper metrics.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsComputer:
    """Unified metrics computation with single definitions."""
    
    # CRITICAL: These definitions are authoritative and used everywhere
    ECE_N_BINS = 10
    ECE_BINNING_SCHEME = "equal_width"  # Equal-width (not equal-frequency)
    ECE_CONFIDENCE_DEF = "predicted_class"  # max(p, 1-p) for binary classification
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize metrics computer.
        
        Args:
            n_bins: Number of bins for ECE (default 10 per Guo et al.)
        """
        self.n_bins = n_bins
        logger.info(f"MetricsComputer initialized: {n_bins} equal-width bins, "
                   f"confidence_def={self.ECE_CONFIDENCE_DEF}")
    
    @staticmethod
    def compute_confidence(probabilities: np.ndarray) -> np.ndarray:
        """
        Compute confidence = max(p, 1-p) for binary classification.
        
        This is the probability of the predicted class.
        
        Args:
            probabilities: Array of shape (n,) with p(SUPPORTED in [0,1])
        
        Returns:
            Confidence array of shape (n,) with max(p, 1-p)
        """
        return np.maximum(probabilities, 1.0 - probabilities)
    
    def compute_ece(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        return_bins: bool = False
    ) -> Dict:
        """
        Compute Expected Calibration Error (ECE).
        
        DEFINITION:
        ECE = sum_k (n_k / N) * |accuracy_k - confidence_k|
        
        where:
        - k indexes bins of equal width [0, 0.1], [0.1, 0.2], ..., [0.9, 1.0]
        - confidence_k = max(p, 1-p) for predicted class
        - accuracy_k = fraction of correct predictions in bin k
        - n_k = count of examples in bin k
        - N = total samples
        
        Args:
            probabilities: Shape (n,), probability of SUPPORTED class
            labels: Shape (n,), 1=SUPPORTED, 0=REFUTED
            return_bins: If True, return per-bin statistics
        
        Returns:
            Dict with 'ece' and optional bin statistics
        """
        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        
        assert len(probabilities) == len(labels)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
        assert np.all((labels == 0) | (labels == 1))
        
        # Compute confidence (max prob)
        confidence = self.compute_confidence(probabilities)
        
        # Compute predicted labels (threshold 0.5)
        predictions = (probabilities >= 0.5).astype(np.int64)
        
        # Compute correctness
        correctness = (predictions == labels).astype(np.int64)
        
        # Create equal-width bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        
        ece = 0.0
        bin_stats = []
        n_total = len(probabilities)
        
        for i in range(self.n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]
            
            # Bin membership (include upper edge only for last bin)
            if i == self.n_bins - 1:
                in_bin = (confidence >= bin_lower) & (confidence <= bin_upper)
            else:
                in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
            
            n_k = np.sum(in_bin)
            
            if n_k > 0:
                acc_k = np.mean(correctness[in_bin])
                conf_k = np.mean(confidence[in_bin])
                ece += (n_k / n_total) * np.abs(acc_k - conf_k)
                
                bin_stats.append({
                    'bin_id': i,
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'count': int(n_k),
                    'accuracy': float(acc_k),
                    'confidence': float(conf_k),
                    'abs_difference': float(np.abs(acc_k - conf_k))
                })
        
        result = {'ece': float(ece)}
        if return_bins:
            result['bins'] = bin_stats
        
        return result
    
    def compute_accuracy_coverage_curve(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute accuracy-coverage curve by varying abstention threshold τ.
        
        DEFINITION:
        For each threshold τ:
        - coverage(τ) = fraction of examples with confidence ≥ τ
        - selective_accuracy(τ) = accuracy among predictions with confidence ≥ τ
        
        Args:
            confidences: Shape (n,), confidence scores in [0, 1]
            correctness: Shape (n,), 1=correct, 0=incorrect
            thresholds: If None, use linspace(0.5, 1.0, 51)
        
        Returns:
            Dict with 'thresholds', 'coverage', 'accuracy' arrays
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        correctness = np.asarray(correctness, dtype=np.int64)
        
        assert len(confidences) == len(correctness)
        assert np.all((confidences >= 0.0) & (confidences <= 1.0))
        
        if thresholds is None:
            thresholds = np.linspace(0.5, 1.0, 51)
        
        thresholds = np.asarray(thresholds, dtype=np.float64)
        
        coverage_list = []
        accuracy_list = []
        
        for tau in thresholds:
            # Predictions with confidence >= tau
            keep_mask = confidences >= tau
            coverage = np.mean(keep_mask)  # Fraction kept
            
            if coverage > 0:
                accuracy = np.mean(correctness[keep_mask])
            else:
                accuracy = 1.0  # No predictions made, no errors
            
            coverage_list.append(float(coverage))
            accuracy_list.append(float(accuracy))
        
        return {
            'thresholds': thresholds.tolist(),
            'coverage': coverage_list,
            'accuracy': accuracy_list
        }
    
    def compute_auc_ac(
        self,
        coverage: np.ndarray,
        accuracy: np.ndarray,
        normalize: bool = True
    ) -> float:
        """
        Compute Area Under Accuracy-Coverage Curve.
        
        DEFINITION:
        AUC-AC = integral of accuracy(coverage) from coverage=0 to coverage=1
                via trapezoidal integration
        
        Normalized: AUC-AC in [0, 1] where 0.5 = random, 1.0 = perfect
        
        Args:
            coverage: Shape (n,), coverage fraction in [0, 1]
            accuracy: Shape (n,), selective accuracy in [0, 1]
            normalize: If True, normalize to [0, 1] range
        
        Returns:
            AUC-AC value
        """
        coverage = np.asarray(coverage, dtype=np.float64)
        accuracy = np.asarray(accuracy, dtype=np.float64)
        
        assert len(coverage) == len(accuracy)
        
        # Sort by coverage for integration
        sort_idx = np.argsort(coverage)
        coverage_sorted = coverage[sort_idx]
        accuracy_sorted = accuracy[sort_idx]
        
        # Trapezoidal integration
        auc = float(np.trapz(accuracy_sorted, coverage_sorted))
        
        if normalize:
            # Normalize to [0, 1]: random selective prediction has AUC-AC ≈ 0.5
            # (since random coverage × random accuracy ≈ 0.5)
            # Perfect has 1.0, worst has 0.0
            auc = auc  # Already in [0, 1] when coverage sorted from 0 to 1
        
        return auc
    
    def compute_all_metrics(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute all metrics at once.
        
        Args:
            probabilities: Shape (n,), p(SUPPORTED)
            labels: Shape (n,), 1=SUPPORTED, 0=REFUTED
            thresholds: Optional thresholds for accuracy-coverage curve
        
        Returns:
            Dict with ece, auc_ac, and per-bin statistics
        """
        # Compute ECE
        ece_result = self.compute_ece(probabilities, labels, return_bins=True)
        ece = ece_result['ece']
        
        # Compute confidence and correctness
        confidence = self.compute_confidence(probabilities)
        predictions = (probabilities >= 0.5).astype(np.int64)
        correctness = (predictions == labels).astype(np.int64)
        
        # Compute accuracy-coverage curve
        curve_result = self.compute_accuracy_coverage_curve(
            confidence, correctness, thresholds
        )
        
        # Compute AUC-AC
        auc_ac = self.compute_auc_ac(
            np.array(curve_result['coverage']),
            np.array(curve_result['accuracy'])
        )
        
        # Compute base accuracy
        accuracy = np.mean(correctness)
        
        # Compute macro-F1 (binary case)
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(labels, predictions, average='binary')
        
        return {
            'accuracy': float(accuracy),
            'ece': float(ece),
            'auc_ac': float(auc_ac),
            'macro_f1': float(macro_f1),
            'ece_bins': ece_result['bins'],
            'accuracy_coverage_curve': {
                'thresholds': curve_result['thresholds'],
                'coverage': curve_result['coverage'],
                'accuracy': curve_result['accuracy']
            },
            'metadata': {
                'n_samples': len(probabilities),
                'ece_n_bins': self.n_bins,
                'ece_binning': self.ECE_BINNING_SCHEME,
                'confidence_definition': self.ECE_CONFIDENCE_DEF
            }
        }


def create_metrics_computer(n_bins: int = 10) -> MetricsComputer:
    """Factory function to create metrics computer."""
    return MetricsComputer(n_bins=n_bins)
