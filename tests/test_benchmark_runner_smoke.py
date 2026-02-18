"""
Smoke test for CSBenchmarkRunner.

Tests that the benchmark runner:
- Loads the dataset correctly
- Runs verification pipeline on a small sample
- Computes reasonable metrics
- Saves output artifacts correctly
"""

import json
import pytest
import tempfile
from pathlib import Path

from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner, BenchmarkMetrics


class TestBenchmarkRunnerSmoke:
    """Smoke tests for benchmark runner - fast validation on small sample."""
    
    @pytest.fixture
    def benchmark_dataset_path(self):
        """Path to CSClaimBench v1.0 dataset."""
        return Path(__file__).parent.parent / "evaluation" / "cs_benchmark" / "csclaimbench_v1.jsonl"
    
    @pytest.fixture
    def runner(self, benchmark_dataset_path):
        """Create benchmark runner with default config."""
        assert benchmark_dataset_path.exists(), f"Benchmark dataset not found: {benchmark_dataset_path}"
        return CSBenchmarkRunner(
            dataset_path=str(benchmark_dataset_path),
            batch_size=4,
            device="cpu",
            seed=42,
            log_predictions=True
        )
    
    def test_dataset_loads(self, runner):
        """Test that dataset loads correctly."""
        assert len(runner.dataset) > 0, "Dataset should have examples"
        assert len(runner.dataset) >= 180, f"Expected at least 180 examples, got {len(runner.dataset)}"
        
        # Check first example has required fields
        example = runner.dataset[0]
        assert "doc_id" in example
        assert "claim" in example
        assert "gold_label" in example
        assert "source_text" in example
        assert "domain_topic" in example
    
    def test_runner_executes_fast_sample(self, runner):
        """Test that runner completes on first 10 examples without errors."""
        result = runner.run(sample_size=10)
        
        assert result is not None, "Runner should return a result"
        assert result.metrics is not None, "Result should contain metrics"
        assert len(result.predictions) == 10, f"Expected 10 predictions, got {len(result.predictions)}"
    
    def test_predictions_format(self, runner):
        """Test that predictions have correct format."""
        result = runner.run(sample_size=5)
        
        for prediction in result.predictions:
            assert "claim_id" in prediction
            assert "pred_label" in prediction
            assert "pred_confidence" in prediction
            assert "gold_label" in prediction
            assert "match" in prediction
            assert "time" in prediction
            assert "evidence_count" in prediction
            
            # Check label values
            assert prediction["pred_label"] in ["ENTAIL", "CONTRADICT", "NEUTRAL"]
            assert prediction["gold_label"] in ["ENTAIL", "CONTRADICT", "NEUTRAL"]
            
            # Check confidence range
            assert 0.0 <= prediction["pred_confidence"] <= 1.0
            
            # Check match is boolean
            assert isinstance(prediction["match"], bool)
    
    def test_metrics_reasonable_ranges(self, runner):
        """Test that computed metrics are in reasonable ranges."""
        result = runner.run(sample_size=10)
        metrics = result.metrics
        
        # Accuracy should be between 0 and 1
        assert 0.0 <= metrics.accuracy <= 1.0, f"Accuracy out of range: {metrics.accuracy}"
        
        # Precision, recall, F1 should be between 0 and 1
        assert 0.0 <= metrics.precision_verified <= 1.0
        assert 0.0 <= metrics.recall_verified <= 1.0
        assert 0.0 <= metrics.F1_verified <= 1.0
        
        assert 0.0 <= metrics.precision_rejected <= 1.0
        assert 0.0 <= metrics.recall_rejected <= 1.0
        assert 0.0 <= metrics.F1_rejected <= 1.0
        
        # ECE should be between 0 and 1
        assert 0.0 <= metrics.ece <= 1.0, f"ECE out of range: {metrics.ece}"
        
        # Brier score should be between 0 and 1
        assert 0.0 <= metrics.brier_score <= 1.0, f"Brier score out of range: {metrics.brier_score}"
        
        # Total claims should match sample size
        assert metrics.total_claims == 10, f"Expected 10 total claims, got {metrics.total_claims}"
        
        # Evidence coverage should be positive
        assert metrics.evidence_coverage_rate >= 0.0, "Evidence coverage rate should be non-negative"
    
    def test_save_artifacts(self, runner):
        """Test that output artifacts can be saved correctly."""
        result = runner.run(sample_size=5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Save predictions as JSONL
            predictions_path = tmpdir_path / "predictions.jsonl"
            with open(predictions_path, 'w') as f:
                for pred in result.predictions:
                    f.write(json.dumps(pred) + '\n')
            
            assert predictions_path.exists(), "Predictions file should be created"
            
            # Verify predictions can be loaded back
            loaded_preds = []
            with open(predictions_path, 'r') as f:
                for line in f:
                    if line.strip():
                        loaded_preds.append(json.loads(line))
            
            assert len(loaded_preds) == 5, f"Expected 5 predictions, loaded {len(loaded_preds)}"
            
            # Save metrics as JSON
            metrics_path = tmpdir_path / "metrics.json"
            result.to_json(metrics_path)
            
            assert metrics_path.exists(), "Metrics JSON should be created"
            
            # Verify metrics can be loaded back
            with open(metrics_path, 'r') as f:
                loaded_metrics = json.load(f)
            
            assert "metrics" in loaded_metrics
            assert "accuracy" in loaded_metrics["metrics"]
            
            # Save metrics as CSV
            csv_path = tmpdir_path / "results.csv"
            result.to_csv(csv_path)
            
            assert csv_path.exists(), "Results CSV should be created"
    
    def test_label_distribution(self, runner):
        """Test that label distribution is tracked correctly."""
        result = runner.run(sample_size=15)
        metrics = result.metrics
        
        # Label distribution should sum to total claims
        label_sum = sum(metrics.label_distribution.values())
        assert label_sum == 15, f"Label distribution should sum to 15, got {label_sum}"
        
        # All labels should be present in distribution
        assert "ENTAIL" in metrics.label_distribution
        assert "CONTRADICT" in metrics.label_distribution
        assert "NEUTRAL" in metrics.label_distribution
    
    def test_evidence_retrieval(self, runner):
        """Test that evidence retrieval works correctly."""
        result = runner.run(sample_size=10)
        
        # At least some claims should have evidence
        claims_with_evidence = sum(1 for p in result.predictions if p["evidence_count"] > 0)
        assert claims_with_evidence > 0, "At least some claims should have retrieved evidence"
        
        # Evidence coverage rate should be reasonable
        coverage_rate = result.metrics.evidence_coverage_rate
        assert coverage_rate > 0.5, f"Evidence coverage rate too low: {coverage_rate}"
    
    def test_timing_metrics(self, runner):
        """Test that timing metrics are recorded correctly."""
        result = runner.run(sample_size=10)
        metrics = result.metrics
        
        # Average time should be positive
        assert metrics.avg_time_per_claim > 0, "Average time per claim should be positive"
        
        # Total time should be reasonable (< 60 seconds for 10 claims)
        assert metrics.total_time < 60.0, f"Total time too high: {metrics.total_time}s"
        
        # Median time should be positive
        assert metrics.median_time_per_claim > 0, "Median time should be positive"
        
        # p95 time should be >= median time
        assert metrics.p95_time_per_claim >= metrics.median_time_per_claim, \
            "p95 time should be >= median time"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
