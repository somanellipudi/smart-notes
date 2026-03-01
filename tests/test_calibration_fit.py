import numpy as np
from src.evaluation.calibration import CalibrationEvaluator


def test_fit_temperature_reduces_ece():
    # Construct synthetic overconfident distribution:
    # High-confidence group (0.9) but only 80% correct
    # Low-confidence group (0.4) but only 20% correct
    high_conf = [0.9] * 50
    low_conf = [0.4] * 50
    probs = high_conf + low_conf

    # Labels: first group 80% ones, second group 20% ones
    labels = [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40

    evaluator = CalibrationEvaluator(n_bins=10)
    original_ece = evaluator.expected_calibration_error(np.array(probs), np.array(labels))

    fit = evaluator.fit_temperature_grid(probs, labels, grid_min=0.5, grid_max=2.0, grid_steps=50)
    assert "best_tau" in fit
    assert "best_ece" in fit

    best_ece = fit["best_ece"]
    assert best_ece <= original_ece + 1e-8
    # Expect a non-trivial reduction in this synthetic scenario
    assert best_ece < original_ece
