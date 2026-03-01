"""
Deterministic unit tests for CalibraTeach evaluation pipeline.

All tests use seed=42 for reproducibility. Synthetic data clearly labeled.

Running: pytest tests/test_evaluation_deterministic.py -v
"""
import pytest
import numpy as np
from pathlib import Path
import json

from src.config.verification_config import VerificationConfig
from src.evaluation.synthetic_data import (
    generate_synthetic_csclaimbench,
    generate_synthetic_calibration_data,
    generate_synthetic_fever_like,
)
from src.evaluation.calibration import CalibrationEvaluator
from src.evaluation.samplers import sample_jsonl_subset
from src.evaluation.plots import (
    plot_reliability_diagram,
    compute_risk_coverage_curve,
    plot_risk_coverage,
)


# ==================== Determinism & Reproducibility Tests ====================


@pytest.mark.unit
def test_synthetic_data_determinism_csclaimbench():
    """Test that synthetic CSClaimBench generation is deterministic."""
    records1 = generate_synthetic_csclaimbench(n_samples=100, seed=42)
    records2 = generate_synthetic_csclaimbench(n_samples=100, seed=42)

    # Same seed should produce identical records
    assert len(records1) == len(records2)
    for r1, r2 in zip(records1, records2):
        assert r1["doc_id"] == r2["doc_id"]
        assert r1["generated_claim"] == r2["generated_claim"]
        assert r1["gold_label"] == r2["gold_label"]


@pytest.mark.unit
def test_synthetic_data_determinism_calibration():
    """Test that synthetic calibration data is deterministic."""
    conf1, labels1 = generate_synthetic_calibration_data(n_samples=100, seed=42)
    conf2, labels2 = generate_synthetic_calibration_data(n_samples=100, seed=42)

    np.testing.assert_array_equal(conf1, conf2)
    np.testing.assert_array_equal(labels1, labels2)


@pytest.mark.unit
def test_synthetic_data_determinism_fever():
    """Test that synthetic FEVER data is deterministic."""
    records1 = generate_synthetic_fever_like(n_samples=50, seed=42)
    records2 = generate_synthetic_fever_like(n_samples=50, seed=42)

    assert len(records1) == len(records2)
    for r1, r2 in zip(records1, records2):
        assert r1 == r2


@pytest.mark.unit
def test_synthetic_data_has_metadata():
    """Test that generated data has synthetic/placeholder metadata."""
    records = generate_synthetic_csclaimbench(n_samples=10, seed=42)

    for rec in records:
        assert "_metadata" in rec
        assert rec["_metadata"]["synthetic"] is True
        assert rec["_metadata"]["placeholder"] is True
        assert rec["_metadata"]["seed"] == 42


# ==================== Config & Deployment Mode Tests ====================


@pytest.mark.unit
def test_config_deployment_mode_full_default(verification_config_full_optimization):
    """Test full_default deployment mode enables all optimizations."""
    cfg = verification_config_full_optimization

    assert cfg.deployment_mode == "full_default"
    assert cfg.enable_result_cache is True
    assert cfg.enable_quality_screening is True
    assert cfg.enable_query_expansion is True
    assert cfg.enable_evidence_ranker is True
    assert cfg.enable_adaptive_depth is True


@pytest.mark.unit
def test_config_deployment_mode_minimal(verification_config_minimal):
    """Test minimal_deployment mode disables most optimizations."""
    cfg = verification_config_minimal

    assert cfg.deployment_mode == "minimal_deployment"
    assert cfg.enable_result_cache is True
    assert cfg.enable_quality_screening is True
    assert cfg.enable_query_expansion is False
    assert cfg.enable_evidence_ranker is False
    assert cfg.enable_type_classifier is False


@pytest.mark.unit
def test_config_deployment_mode_verifiable(verification_config):
    """Test verifiable mode disables all optimizations."""
    cfg = verification_config

    assert cfg.deployment_mode == "verifiable"
    assert cfg.enable_result_cache is False
    assert cfg.enable_quality_screening is False
    assert cfg.enable_query_expansion is False


@pytest.mark.unit
def test_config_as_dict():
    """Test that config serializes to dict correctly."""
    cfg = VerificationConfig(random_seed=42)
    d = cfg.as_dict()

    assert isinstance(d, dict)
    assert d["random_seed"] == 42
    assert "deployment_mode" in d
    assert "enable_result_cache" in d


@pytest.mark.unit
def test_config_invalid_deployment_mode():
    """Test that invalid deployment mode raises error."""
    with pytest.raises(Exception):
        VerificationConfig(deployment_mode="invalid_mode")  # type: ignore


# ==================== Calibration Tests ====================


@pytest.mark.unit
def test_calibration_evaluator_basic(synthetic_calibration_data):
    """Test calibration evaluator computes metrics."""
    confidences, labels = synthetic_calibration_data

    evaluator = CalibrationEvaluator(n_bins=10)
    metrics = evaluator.evaluate(confidences, labels, return_bins=False)

    assert "ece" in metrics
    assert "brier_score" in metrics
    assert "accuracy" in metrics
    assert "n_samples" in metrics
    assert metrics["n_samples"] == 100
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0


@pytest.mark.unit
def test_calibration_evaluator_perfectly_calibrated():
    """Test calibration evaluator on nearly-perfect calibration."""
    np.random.seed(42)
    # Create perfectly calibrated data: confidence == accuracy
    n = 1000
    confidences = np.random.uniform(0, 1, n)
    labels = (np.random.uniform(0, 1, n) < confidences).astype(int)

    evaluator = CalibrationEvaluator(n_bins=10)
    metrics = evaluator.evaluate(confidences, labels, return_bins=True)

    # ECE should be relatively low (well-calibrated)
    assert metrics["ece"] < 0.15


@pytest.mark.unit
def test_calibration_evaluator_with_bins():
    """Test calibration evaluator with bin statistics."""
    confidences, labels = generate_synthetic_calibration_data(n_samples=100, seed=42)

    evaluator = CalibrationEvaluator(n_bins=5)
    metrics = evaluator.evaluate(confidences, labels, return_bins=True)

    assert "bins" in metrics
    assert len(metrics["bins"]) == 5


# ==================== Sampler Tests ====================


@pytest.mark.unit
def test_jsonl_sampler_determinism(synthetic_csclaimbench_jsonl):
    """Test that JSONL sampler is deterministic."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        out1 = Path(tmpdir) / "sample1.jsonl"
        out2 = Path(tmpdir) / "sample2.jsonl"

        lines1 = sample_jsonl_subset(synthetic_csclaimbench_jsonl, str(out1), n=50, seed=42)
        lines2 = sample_jsonl_subset(synthetic_csclaimbench_jsonl, str(out2), n=50, seed=42)

        assert len(lines1) == len(lines2)
        assert lines1 == lines2


@pytest.mark.unit
def test_jsonl_sampler_writes_file(synthetic_csclaimbench_jsonl):
    """Test that JSONL sampler writes output file."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "sampled.jsonl"
        lines = sample_jsonl_subset(synthetic_csclaimbench_jsonl, str(outpath), n=30, seed=42)

        assert outpath.exists()
        assert len(lines) == 30

        # Verify JSONL format
        with outpath.open() as f:
            for line in f:
                json.loads(line)


# ==================== Plotting Tests ====================


@pytest.mark.unit
def test_reliability_diagram_plot(synthetic_calibration_data, temp_output_dir):
    """Test reliability diagram plotting."""
    confidences, labels = synthetic_calibration_data

    outpath = temp_output_dir / "reliability.png"
    plot_reliability_diagram(confidences, labels, str(outpath), num_bins=10)

    assert outpath.exists()
    assert outpath.stat().st_size > 0


@pytest.mark.unit
def test_risk_coverage_curve_computation(synthetic_calibration_data):
    """Test risk-coverage curve computation."""
    confidences, labels = synthetic_calibration_data

    # Create predictions (same as labels for simplicity)
    predictions = labels.copy()

    thresholds, coverage, accuracy = compute_risk_coverage_curve(
        confidences, predictions, labels, n_points=50
    )

    assert len(thresholds) == 50
    assert len(coverage) == 50
    assert len(accuracy) == 50

    # Coverage should be monotonically decreasing with threshold
    assert coverage[0] >= coverage[-1]

    # Accuracy typically increases with coverage (more selective = more accurate)
    # But this varies based on data


@pytest.mark.unit
def test_risk_coverage_plot(synthetic_calibration_data, temp_output_dir):
    """Test risk-coverage curve plotting."""
    confidences, labels = synthetic_calibration_data
    predictions = labels.copy()

    outpath = temp_output_dir / "risk_coverage.png"
    plot_risk_coverage(confidences, predictions, labels, str(outpath))

    assert outpath.exists()
    assert outpath.stat().st_size > 0


# ==================== Integration Tests ====================


@pytest.mark.unit
def test_synthetic_pipeline_shapes():
    """Test synthetic data shapes for full pipeline."""
    n_csc = 300
    n_fever = 200

    data_csc = generate_synthetic_csclaimbench(n_samples=n_csc, seed=42)
    data_fever = generate_synthetic_fever_like(n_samples=n_fever, seed=42)
    conf, labels = generate_synthetic_calibration_data(n_samples=100, seed=42)

    assert len(data_csc) == n_csc
    assert len(data_fever) == n_fever
    assert len(conf) == 100
    assert len(labels) == 100


@pytest.mark.unit
def test_end_to_end_synthetic_workflow(temp_output_dir):
    """Test end-to-end synthetic workflow: generate -> calibrate -> plot."""
    # 1. Generate synthetic data
    csc_path = temp_output_dir / "csc_synth.jsonl"
    generate_synthetic_csclaimbench(n_samples=100, seed=42, outpath=str(csc_path))
    assert csc_path.exists()

    # 2. Generate calibration data and calibrate
    confidences, labels = generate_synthetic_calibration_data(n_samples=100, seed=42)
    evaluator = CalibrationEvaluator(n_bins=10)
    metrics = evaluator.evaluate(confidences, labels)
    assert "ece" in metrics

    # 3. Plot reliability diagram
    plot_path = temp_output_dir / "reliability.png"
    plot_reliability_diagram(confidences, labels, str(plot_path))
    assert plot_path.exists()

    print(f"✓ End-to-end workflow completed. ECE={metrics['ece']:.4f}")


# ==================== Reproducibility Stress Tests ====================


@pytest.mark.slow
@pytest.mark.unit
def test_large_scale_determinism():
    """Test determinism on larger dataset (stress test)."""
    # Generate large synthetic dataset
    records1 = generate_synthetic_csclaimbench(n_samples=1000, seed=42)
    records2 = generate_synthetic_csclaimbench(n_samples=1000, seed=42)

    assert len(records1) == len(records2) == 1000

    # Check first, middle, and last records are identical
    for i in [0, 500, 999]:
        assert records1[i] == records2[i]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
