import numpy as np

from src.eval.metrics import (
    compute_accuracy_coverage_curve,
    compute_auc_ac,
    compute_ece,
)


def test_ece_zero_for_perfect_calibration_toy_data():
    y_true = np.array([1, 1, 0, 0], dtype=np.int64)
    probs = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)

    result = compute_ece(y_true, probs, n_bins=10, scheme="equal_width", confidence_mode="predicted_class")
    assert abs(result["ece"]) < 1e-12


def test_ece_increases_when_overconfident():
    y_true = np.array([1, 1, 0, 0], dtype=np.int64)

    # 50% correct in both settings, but higher confidence should increase ECE.
    calibrated_probs = np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float64)
    overconfident_probs = np.array([0.95, 0.95, 0.95, 0.95], dtype=np.float64)

    ece_cal = compute_ece(y_true, calibrated_probs)["ece"]
    ece_over = compute_ece(y_true, overconfident_probs)["ece"]

    assert ece_over >= ece_cal


def test_auc_ac_matches_hand_computed_trapezoid():
    coverage = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    accuracy = np.array([1.0, 0.8, 0.6], dtype=np.float64)

    expected = ((1.0 + 0.8) * 0.5 / 2.0) + ((0.8 + 0.6) * 0.5 / 2.0)
    got = compute_auc_ac(coverage, accuracy)

    assert abs(got - expected) < 1e-12


def test_coverage_monotonic_sanity():
    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)
    probs = np.array([0.9, 0.2, 0.8, 0.4, 0.7, 0.3], dtype=np.float64)

    curve = compute_accuracy_coverage_curve(
        y_true,
        probs,
        confidence_mode="predicted_class",
        thresholds="unique",
    )

    coverage = np.array(curve["coverage"], dtype=np.float64)
    assert np.all(np.diff(coverage) >= -1e-12)
    assert np.all((coverage >= 0.0) & (coverage <= 1.0))
