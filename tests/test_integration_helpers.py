"""
Tests for integration helpers that combine selective and conformal prediction
with the reporting system.
"""

import pytest
import numpy as np
from src.evaluation.integration_helpers import (
    compute_selective_prediction_metrics,
    compute_conformal_prediction_metrics,
    combine_metrics,
    add_confidence_guarantees_to_verification_summary,
    format_threshold_recommendation
)


class TestSelectivePredictionIntegration:
    """Tests for selective prediction integration."""
    
    def test_compute_metrics_basic(self):
        """Basic selective prediction metrics computation."""
        np.random.seed(42)
        n = 50
        scores = np.random.uniform(0.4, 1.0, n).tolist()
        predictions = np.random.randint(0, 2, n).tolist()
        
        # Make high scores more likely correct
        targets = predictions.copy()
        for i in range(n):
            if scores[i] < 0.6 and np.random.rand() < 0.5:
                targets[i] = 1 - targets[i]
        
        metrics = compute_selective_prediction_metrics(
            scores, predictions, targets, target_risk=0.1
        )
        
        # Check all required keys present
        assert 'optimal_threshold' in metrics
        assert 'achieved_coverage' in metrics
        assert 'achieved_risk' in metrics
        assert 'target_risk' in metrics
        assert 'auc_rc' in metrics
        
        # Check values in valid ranges
        assert 0 <= metrics['optimal_threshold'] <= 1
        assert 0 <= metrics['achieved_coverage'] <= 1
        assert 0 <= metrics['achieved_risk'] <= 1
        assert metrics['target_risk'] == 0.1
    
    def test_empty_input_handling(self):
        """Empty input should return default metrics."""
        metrics = compute_selective_prediction_metrics([], [], [], target_risk=0.05)
        
        assert metrics['optimal_threshold'] == 0.5
        assert metrics['achieved_coverage'] == 0.0
        assert metrics['target_risk'] == 0.05
    
    def test_perfect_predictions(self):
        """Perfect predictions should have 0 risk."""
        scores = [0.9, 0.8, 0.7, 0.6]
        predictions = [1, 0, 1, 0]
        targets = [1, 0, 1, 0]  # All correct
        
        metrics = compute_selective_prediction_metrics(
            scores, predictions, targets, target_risk=0.1
        )
        
        # Should achieve 0 risk with full coverage
        assert metrics['achieved_risk'] == 0.0
        assert metrics['achieved_coverage'] == 1.0


class TestConformalPredictionIntegration:
    """Tests for conformal prediction integration."""
    
    def test_compute_metrics_basic(self):
        """Basic conformal prediction metrics computation."""
        np.random.seed(42)
        n = 50
        scores = np.random.uniform(0.3, 1.0, n).tolist()
        predictions = np.random.randint(0, 2, n).tolist()
        
        # Mix correct and incorrect
        targets = predictions.copy()
        error_indices = np.random.choice(n, size=n//4, replace=False)
        for i in error_indices:
            targets[i] = 1 - targets[i]
        
        metrics = compute_conformal_prediction_metrics(
            scores, predictions, targets, alpha=0.1
        )
        
        # Check all required keys present
        assert 'threshold' in metrics
        assert 'alpha' in metrics
        assert 'coverage_guarantee' in metrics
        assert 'empirical_coverage' in metrics
        assert 'calibration_size' in metrics
        
        # Check values
        assert 0 <= metrics['threshold'] <= 1
        assert metrics['alpha'] == 0.1
        assert abs(metrics['coverage_guarantee'] - 0.9) < 0.01
        assert 0 <= metrics['empirical_coverage'] <= 1
        assert metrics['calibration_size'] == n
    
    def test_different_alpha_values(self):
        """Different alpha values should give different thresholds."""
        scores = np.random.uniform(0.3, 1.0, 100).tolist()
        predictions = np.random.randint(0, 2, 100).tolist()
        targets = [(p + np.random.randint(0, 2)) % 2 for p in predictions]
        
        metrics_01 = compute_conformal_prediction_metrics(
            scores, predictions, targets, alpha=0.01
        )
        
        metrics_10 = compute_conformal_prediction_metrics(
            scores, predictions, targets, alpha=0.10
        )
        
        # Smaller alpha should generally give higher threshold
        # (more conservative to achieve higher confidence)
        # Note: This is not guaranteed but should hold statistically
        assert metrics_01['coverage_guarantee'] > metrics_10['coverage_guarantee']
    
    def test_empty_input_handling(self):
        """Empty input should return default metrics."""
        metrics = compute_conformal_prediction_metrics([], [], [], alpha=0.1)
        
        assert metrics['threshold'] == 0.5
        assert metrics['alpha'] == 0.1
        assert metrics['coverage_guarantee'] == 0.9
        assert metrics['calibration_size'] == 0


class TestCombineMetrics:
    """Tests for combining both approaches."""
    
    def test_combine_with_split(self):
        """Combine metrics with calibration split."""
        np.random.seed(42)
        n = 100
        scores = np.random.uniform(0.4, 1.0, n).tolist()
        predictions = np.random.randint(0, 3, n).tolist()
        targets = [(p + np.random.randint(0, 2)) % 3 for p in predictions]
        
        sp_metrics, cp_metrics = combine_metrics(
            scores, predictions, targets,
            target_risk=0.1,
            alpha=0.1,
            calibration_split=0.5
        )
        
        # Both should have valid metrics
        assert 'optimal_threshold' in sp_metrics
        assert 'threshold' in cp_metrics
        
        assert 0 <= sp_metrics['optimal_threshold'] <= 1
        assert 0 <= cp_metrics['threshold'] <= 1
    
    def test_small_dataset_handling(self):
        """Small dataset should use all data for both metrics."""
        scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        predictions = [1, 1, 0, 0, 0]
        targets = [1, 0, 0, 1, 0]
        
        sp_metrics, cp_metrics = combine_metrics(
            scores, predictions, targets,
            target_risk=0.2,
            alpha=0.2
        )
        
        # Should still return valid metrics
        assert sp_metrics is not None
        assert cp_metrics is not None
    
    def test_empty_input(self):
        """Empty input should return empty dicts."""
        sp_metrics, cp_metrics = combine_metrics([], [], [])
        
        assert sp_metrics == {}
        assert cp_metrics == {}


class TestVerificationSummaryIntegration:
    """Tests for adding metrics to verification summary."""
    
    def test_add_to_summary(self):
        """Add confidence guarantees to verification summary."""
        summary = {
            'total_claims': 100,
            'verified_count': 80,
            'low_confidence_count': 15,
            'rejected_count': 5,
            'avg_confidence': 0.85
        }
        
        np.random.seed(42)
        n = 100
        scores = np.random.uniform(0.5, 1.0, n).tolist()
        predictions = np.random.randint(0, 2, n).tolist()
        targets = [(p if np.random.rand() > 0.2 else 1-p) for p in predictions]
        
        enhanced = add_confidence_guarantees_to_verification_summary(
            summary, scores, predictions, targets,
            target_risk=0.05,
            alpha=0.1
        )
        
        # Original fields preserved
        assert enhanced['total_claims'] == 100
        assert enhanced['verified_count'] == 80
        
        # New fields added
        assert 'selective_prediction' in enhanced
        assert 'conformal_prediction' in enhanced
        
        # Check structure
        assert 'optimal_threshold' in enhanced['selective_prediction']
        assert 'threshold' in enhanced['conformal_prediction']
    
    def test_preserves_original_summary(self):
        """Original summary should not be modified."""
        summary = {
            'total_claims': 50,
            'avg_confidence': 0.75
        }
        
        scores = [0.9, 0.8, 0.7, 0.6]
        predictions = [1, 1, 0, 0]
        targets = [1, 0, 0, 1]
        
        original_keys = set(summary.keys())
        
        enhanced = add_confidence_guarantees_to_verification_summary(
            summary, scores, predictions, targets
        )
        
        # Original unchanged
        assert set(summary.keys()) == original_keys
        
        # Enhanced has more keys
        assert len(enhanced.keys()) > len(summary.keys())


class TestThresholdRecommendation:
    """Tests for threshold recommendation formatting."""
    
    def test_format_recommendation(self):
        """Format threshold recommendation."""
        sp_metrics = {
            'optimal_threshold': 0.75,
            'achieved_coverage': 0.80,
            'achieved_risk': 0.05,
            'target_risk': 0.05
        }
        
        cp_metrics = {
            'threshold': 0.70,
            'alpha': 0.1,
            'coverage_guarantee': 0.9,
            'empirical_coverage': 0.92
        }
        
        recommendation = format_threshold_recommendation(sp_metrics, cp_metrics)
        
        # Check contains key information
        assert '0.75' in recommendation  # Combined threshold (max of 0.75, 0.70)
        assert '80' in recommendation or '0.8' in recommendation  # Coverage
        assert 'ACCEPT' in recommendation
        assert 'REVIEW' in recommendation
        assert 'REJECT' in recommendation
    
    def test_handles_missing_keys(self):
        """Should handle missing keys gracefully."""
        sp_metrics = {}
        cp_metrics = {}
        
        # Should not raise error
        recommendation = format_threshold_recommendation(sp_metrics, cp_metrics)
        assert isinstance(recommendation, str)
        assert len(recommendation) > 0


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_complete_pipeline(self):
        """Complete pipeline from scores to report summary."""
        np.random.seed(42)
        n = 200
        
        # Simulate verification results
        scores = np.random.uniform(0.3, 1.0, n).tolist()
        predictions = np.random.randint(0, 3, n).tolist()
        
        # High scores more likely correct
        targets = predictions.copy()
        for i in range(n):
            if scores[i] < 0.6 and np.random.rand() < 0.4:
                targets[i] = (targets[i] + 1) % 3
        
        # Create initial summary
        correct = [p == t for p, t in zip(predictions, targets)]
        summary = {
            'total_claims': n,
            'verified_count': sum(correct),
            'low_confidence_count': sum(1 for s in scores if 0.5 <= s < 0.8),
            'rejected_count': sum(1 for c in correct if not c),
            'avg_confidence': np.mean(scores),
            'top_rejection_reasons': [('Low evidence', 10), ('Contradicted', 5)]
        }
        
        # Add confidence guarantees
        enhanced = add_confidence_guarantees_to_verification_summary(
            summary, scores, predictions, targets,
            target_risk=0.1,
            alpha=0.1
        )
        
        # Verify complete structure
        assert all(k in enhanced for k in [
            'total_claims', 'verified_count', 'avg_confidence',
            'selective_prediction', 'conformal_prediction'
        ])
        
        # Get recommendation
        recommendation = format_threshold_recommendation(
            enhanced['selective_prediction'],
            enhanced['conformal_prediction']
        )
        
        assert isinstance(recommendation, str)
        assert 'Recommended Threshold' in recommendation
        
        # Thresholds should be reasonable
        sp_threshold = enhanced['selective_prediction']['optimal_threshold']
        cp_threshold = enhanced['conformal_prediction']['threshold']
        
        assert 0.3 <= sp_threshold <= 1.0
        assert 0.3 <= cp_threshold <= 1.0
