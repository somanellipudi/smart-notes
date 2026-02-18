"""
Tests for Selective Prediction Module

Tests risk-coverage curves, threshold selection, and AUC computation.
"""

import pytest
import numpy as np
from src.evaluation.selective_prediction import (
    compute_risk_coverage_curve,
    find_threshold_for_target_risk,
    compute_auc_risk_coverage,
    selective_prediction_analysis,
    apply_selective_prediction,
    get_risk_for_coverage,
    get_coverage_for_risk,
    format_selective_prediction_summary,
    RiskCoveragePoint,
    SelectivePredictionResult
)


class TestRiskCoverageCurve:
    """Tests for risk-coverage curve computation."""
    
    def test_perfect_predictions(self):
        """All predictions correct → risk should be 0 at all coverages."""
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        predictions = np.array([1, 0, 1, 0])
        targets = np.array([1, 0, 1, 0])  # All correct
        
        curve = compute_risk_coverage_curve(scores, predictions, targets, num_thresholds=10)
        
        # All points should have risk = 0
        for point in curve:
            if point.num_accepted > 0:
                assert point.risk == 0.0
                assert point.num_errors == 0
    
    def test_all_errors(self):
        """All predictions wrong → risk should be 1.0 at all coverages."""
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        predictions = np.array([1, 1, 1, 1])
        targets = np.array([0, 0, 0, 0])  # All wrong
        
        curve = compute_risk_coverage_curve(scores, predictions, targets, num_thresholds=10)
        
        # All points with coverage > 0 should have risk = 1.0
        for point in curve:
            if point.num_accepted > 0:
                assert point.risk == 1.0
                assert point.num_correct == 0
    
    def test_mixed_predictions(self):
        """Mixed correct/incorrect predictions."""
        scores = np.array([0.9, 0.8, 0.6, 0.4])
        predictions = np.array([1, 1, 0, 0])
        targets = np.array([1, 0, 0, 1])  # 1st and 3rd correct
        
        curve = compute_risk_coverage_curve(scores, predictions, targets, num_thresholds=20)
        
        # Check that curve is sorted by decreasing coverage
        coverages = [p.coverage for p in curve]
        assert coverages == sorted(coverages, reverse=True)
        
        # Check some specific points
        # At high threshold (accept only most confident)
        high_threshold_point = curve[-1]  # Lowest coverage
        assert high_threshold_point.coverage <= 0.5  # Accepts few
        
        # At low threshold (accept all)
        low_threshold_point = curve[0]  # Highest coverage
        assert low_threshold_point.coverage == 1.0
        assert low_threshold_point.num_accepted == 4
        assert low_threshold_point.risk == 0.5  # 2 errors out of 4
    
    def test_selective_improvement(self):
        """Higher confidence predictions should have lower risk."""
        # High confidence correct, low confidence errors
        scores = np.array([0.95, 0.9, 0.85, 0.4, 0.3, 0.2])
        predictions = np.array([1, 1, 1, 0, 0, 0])
        targets = np.array([1, 1, 1, 1, 1, 1])  # First 3 correct, last 3 wrong
        
        curve = compute_risk_coverage_curve(scores, predictions, targets, num_thresholds=10)
        
        # Find point accepting only top 3 (coverage = 0.5)
        top_half_points = [p for p in curve if 0.4 <= p.coverage <= 0.6]
        assert len(top_half_points) > 0
        
        # Risk should be 0 for top 3
        for point in top_half_points:
            if point.num_accepted == 3:
                assert point.risk == 0.0
    
    def test_empty_input(self):
        """Empty arrays should return empty curve."""
        scores = np.array([])
        predictions = np.array([])
        targets = np.array([])
        
        curve = compute_risk_coverage_curve(scores, predictions, targets)
        assert len(curve) == 0
    
    def test_single_example(self):
        """Single example should work."""
        scores = np.array([0.8])
        predictions = np.array([1])
        targets = np.array([1])
        
        curve = compute_risk_coverage_curve(scores, predictions, targets, num_thresholds=5)
        
        # Should have points for coverage 0 and 1
        assert any(p.coverage == 0.0 for p in curve)
        assert any(p.coverage == 1.0 for p in curve)
    
    def test_mismatched_lengths(self):
        """Mismatched input lengths should raise error."""
        scores = np.array([0.9, 0.8])
        predictions = np.array([1, 0, 1])  # Different length
        targets = np.array([1, 0])
        
        with pytest.raises(ValueError, match="must have same length"):
            compute_risk_coverage_curve(scores, predictions, targets)
    
    def test_coverage_range(self):
        """Coverage should be between 0 and 1."""
        scores = np.array([0.9, 0.7, 0.5, 0.3])
        predictions = np.array([1, 1, 0, 0])
        targets = np.array([1, 0, 0, 1])
        
        curve = compute_risk_coverage_curve(scores, predictions, targets)
        
        for point in curve:
            assert 0.0 <= point.coverage <= 1.0
    
    def test_risk_range(self):
        """Risk should be between 0 and 1."""
        scores = np.array([0.9, 0.7, 0.5, 0.3])
        predictions = np.array([1, 1, 0, 0])
        targets = np.array([1, 0, 0, 1])
        
        curve = compute_risk_coverage_curve(scores, predictions, targets)
        
        for point in curve:
            assert 0.0 <= point.risk <= 1.0


class TestThresholdSelection:
    """Tests for finding optimal threshold."""
    
    def test_achievable_target_risk(self):
        """Target risk is achievable → should find valid threshold."""
        # Create simple curve
        curve = [
            RiskCoveragePoint(0.9, 0.2, 0.0, 10, 10, 0),
            RiskCoveragePoint(0.7, 0.5, 0.1, 25, 22, 3),
            RiskCoveragePoint(0.5, 0.8, 0.3, 40, 28, 12),
            RiskCoveragePoint(0.3, 1.0, 0.5, 50, 25, 25),
        ]
        
        threshold, risk, coverage = find_threshold_for_target_risk(curve, target_risk=0.15)
        
        # Should select point with risk=0.1 (highest coverage satisfying target)
        assert risk <= 0.15
        assert coverage == 0.5
        assert threshold == 0.7
    
    def test_unachievable_target_risk(self):
        """Target risk too strict → should return best available."""
        curve = [
            RiskCoveragePoint(0.7, 0.5, 0.2, 25, 20, 5),
            RiskCoveragePoint(0.5, 0.8, 0.3, 40, 28, 12),
            RiskCoveragePoint(0.3, 1.0, 0.5, 50, 25, 25),
        ]
        
        threshold, risk, coverage = find_threshold_for_target_risk(curve, target_risk=0.05)
        
        # Should return point with lowest risk (0.2)
        assert risk == 0.2
        assert coverage == 0.5
    
    def test_prefer_higher_coverage(self):
        """Multiple thresholds achieve target → prefer higher coverage."""
        curve = [
            RiskCoveragePoint(0.9, 0.3, 0.05, 15, 14, 1),
            RiskCoveragePoint(0.8, 0.5, 0.08, 25, 23, 2),
            RiskCoveragePoint(0.7, 0.8, 0.09, 40, 36, 4),
        ]
        
        threshold, risk, coverage = find_threshold_for_target_risk(
            curve, target_risk=0.1, prefer_higher_coverage=True
        )
        
        # Should select highest coverage point with risk <= 0.1
        assert risk <= 0.1
        assert coverage == 0.8  # Highest coverage
        assert threshold == 0.7
    
    def test_empty_curve(self):
        """Empty curve should raise error."""
        with pytest.raises(ValueError, match="Empty"):
            find_threshold_for_target_risk([], target_risk=0.1)
    
    def test_exact_target_match(self):
        """Exact match of target risk."""
        curve = [
            RiskCoveragePoint(0.8, 0.5, 0.1, 25, 22, 3),
            RiskCoveragePoint(0.6, 0.8, 0.2, 40, 32, 8),
        ]
        
        threshold, risk, coverage = find_threshold_for_target_risk(curve, target_risk=0.1)
        
        assert risk == 0.1
        assert coverage == 0.5


class TestAUCComputation:
    """Tests for AUC-RC computation."""
    
    def test_perfect_predictions_low_auc(self):
        """Perfect predictions → AUC should be 0."""
        curve = [
            RiskCoveragePoint(0.0, 0.0, 0.0, 0, 0, 0),
            RiskCoveragePoint(0.5, 0.5, 0.0, 50, 50, 0),
            RiskCoveragePoint(0.1, 1.0, 0.0, 100, 100, 0),
        ]
        
        auc = compute_auc_risk_coverage(curve)
        assert auc == 0.0
    
    def test_all_errors_high_auc(self):
        """All errors → AUC should be 1.0."""
        curve = [
            RiskCoveragePoint(0.0, 0.0, 1.0, 0, 0, 0),
            RiskCoveragePoint(0.5, 0.5, 1.0, 50, 0, 50),
            RiskCoveragePoint(0.1, 1.0, 1.0, 100, 0, 100),
        ]
        
        auc = compute_auc_risk_coverage(curve)
        assert auc == 1.0
    
    def test_linear_tradeoff(self):
        """Linear risk-coverage tradeoff."""
        curve = [
            RiskCoveragePoint(0.9, 0.0, 0.5, 0, 0, 0),
            RiskCoveragePoint(0.5, 0.5, 0.25, 50, 37, 13),
            RiskCoveragePoint(0.1, 1.0, 0.0, 100, 100, 0),
        ]
        
        auc = compute_auc_risk_coverage(curve)
        
        # Trapezoid: (0.5-0)*(0.5+0.25)/2 + (1.0-0.5)*(0.25+0)/2
        # = 0.5*0.375 + 0.5*0.125 = 0.1875 + 0.0625 = 0.25
        assert abs(auc - 0.25) < 0.01
    
    def test_empty_curve(self):
        """Empty curve → AUC should be 0."""
        assert compute_auc_risk_coverage([]) == 0.0
    
    def test_single_point(self):
        """Single point → AUC should be 0."""
        curve = [RiskCoveragePoint(0.5, 0.5, 0.2, 50, 40, 10)]
        assert compute_auc_risk_coverage(curve) == 0.0


class TestSelectivePredictionAnalysis:
    """Tests for full selective prediction analysis."""
    
    def test_complete_analysis(self):
        """Complete selective prediction analysis."""
        np.random.seed(42)
        n = 100
        
        # Simulate scores and predictions
        scores = np.random.uniform(0.3, 1.0, n)
        predictions = np.random.randint(0, 2, n)
        
        # Make high-confidence predictions more likely correct
        targets = predictions.copy()
        # Introduce errors in low-confidence predictions
        low_conf = scores < 0.6
        error_mask = low_conf & (np.random.rand(n) < 0.3)
        targets[error_mask] = 1 - targets[error_mask]
        
        result = selective_prediction_analysis(
            scores, predictions, targets, target_risk=0.1, num_thresholds=50
        )
        
        # Check result structure
        assert isinstance(result, SelectivePredictionResult)
        assert len(result.risk_coverage_curve) > 0
        assert 0 <= result.optimal_threshold <= 1
        assert result.target_risk == 0.1
        assert 0 <= result.achieved_risk <= 1
        assert 0 <= result.achieved_coverage <= 1
        assert result.auc_rc >= 0
        
        # Achieved risk should be close to or better than target
        assert result.achieved_risk <= result.target_risk * 1.5  # Allow some slack
    
    def test_analysis_with_perfect_predictions(self):
        """Analysis with perfect predictions."""
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        predictions = np.array([1, 0, 1, 0])
        targets = predictions.copy()
        
        result = selective_prediction_analysis(
            scores, predictions, targets, target_risk=0.05
        )
        
        # Should achieve 0 risk with full coverage
        assert result.achieved_risk == 0.0
        assert result.achieved_coverage == 1.0
        assert result.auc_rc == 0.0


class TestApplySelectivePrediction:
    """Tests for applying selective prediction."""
    
    def test_apply_threshold(self):
        """Apply threshold to accept/reject predictions."""
        scores = np.array([0.9, 0.7, 0.5, 0.3])
        predictions = np.array([1, 1, 0, 0])
        threshold = 0.6
        
        accepted, rejected = apply_selective_prediction(scores, predictions, threshold)
        
        assert np.array_equal(accepted, np.array([0, 1]))
        assert np.array_equal(rejected, np.array([2, 3]))
    
    def test_accept_all(self):
        """Threshold below all scores → accept all."""
        scores = np.array([0.9, 0.8, 0.7])
        predictions = np.array([1, 0, 1])
        threshold = 0.5
        
        accepted, rejected = apply_selective_prediction(scores, predictions, threshold)
        
        assert len(accepted) == 3
        assert len(rejected) == 0
    
    def test_reject_all(self):
        """Threshold above all scores → reject all."""
        scores = np.array([0.5, 0.4, 0.3])
        predictions = np.array([1, 0, 1])
        threshold = 0.9
        
        accepted, rejected = apply_selective_prediction(scores, predictions, threshold)
        
        assert len(accepted) == 0
        assert len(rejected) == 3


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_risk_for_coverage(self):
        """Get risk at specific coverage level."""
        curve = [
            RiskCoveragePoint(0.9, 0.2, 0.0, 10, 10, 0),
            RiskCoveragePoint(0.7, 0.5, 0.1, 25, 22, 3),
            RiskCoveragePoint(0.5, 0.8, 0.3, 40, 28, 12),
        ]
        
        # At coverage 0.5, risk should be 0.1
        risk = get_risk_for_coverage(curve, target_coverage=0.5)
        assert risk == 0.1
    
    def test_get_coverage_for_risk(self):
        """Get coverage at specific risk level."""
        curve = [
            RiskCoveragePoint(0.9, 0.2, 0.0, 10, 10, 0),
            RiskCoveragePoint(0.7, 0.5, 0.1, 25, 22, 3),
            RiskCoveragePoint(0.5, 0.8, 0.3, 40, 28, 12),
        ]
        
        # At risk 0.1, max coverage should be 0.5
        coverage = get_coverage_for_risk(curve, target_risk=0.1)
        assert coverage == 0.5
    
    def test_format_summary(self):
        """Format summary string."""
        result = SelectivePredictionResult(
            risk_coverage_curve=[],
            optimal_threshold=0.75,
            target_risk=0.1,
            achieved_risk=0.08,
            achieved_coverage=0.65,
            auc_rc=0.12
        )
        
        summary = format_selective_prediction_summary(result)
        
        assert "0.75" in summary
        assert "0.1" in summary or "10" in summary
        assert "65" in summary or "0.65" in summary


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_all_same_score(self):
        """All examples have same score."""
        scores = np.array([0.7, 0.7, 0.7, 0.7])
        predictions = np.array([1, 0, 1, 0])
        targets = np.array([1, 0, 0, 1])
        
        curve = compute_risk_coverage_curve(scores, predictions, targets)
        
        # Should have curve points
        assert len(curve) > 0
    
    def test_target_risk_zero(self):
        """Target risk of 0 (only accept perfect predictions)."""
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        predictions = np.array([1, 0, 1, 0])
        targets = np.array([1, 0, 0, 1])  # 1st and 2nd correct
        
        threshold, risk, coverage = find_threshold_for_target_risk(
            compute_risk_coverage_curve(scores, predictions, targets),
            target_risk=0.0
        )
        
        # Should select only examples with 0 error
        assert risk <= 0.01  # Allow for floating point
    
    def test_target_risk_one(self):
        """Target risk of 1.0 (accept all)."""
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        predictions = np.array([1, 0, 1, 0])
        targets = np.array([0, 1, 0, 1])  # All wrong
        
        threshold, risk, coverage = find_threshold_for_target_risk(
            compute_risk_coverage_curve(scores, predictions, targets),
            target_risk=1.0
        )
        
        # Should accept all
        assert coverage == 1.0
