"""
Smoke test for ingestion noise robustness evaluation.

Tests that CSBenchmarkRunner can:
- Run evaluation on clean dataset
- Apply ingestion noise types (headers/footers, OCR errors, column shuffle)
- Report degradation metrics
- Complete without errors on small samples
"""

import pytest
from pathlib import Path

from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner


class TestRobustnessRunnerSmoke:
    """Smoke tests for robustness evaluation."""
    
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
            log_predictions=False  # Reduce output
        )
    
    def test_run_ingestion_robustness_completes(self, runner):
        """Test that ingestion robustness eval completes without errors."""
        # Run on very small sample for speed
        results = runner.run_ingestion_robustness_eval(
            sample_size=5,
            noise_types=["headers_footers", "ocr_typos"]
        )
        
        assert results is not None, "Should return results"
        assert "clean" in results, "Should have clean baseline"
        assert "headers_footers" in results, "Should have headers_footers variant"
        assert "ocr_typos" in results, "Should have ocr_typos variant"
    
    def test_clean_baseline_present(self, runner):
        """Test that clean baseline is evaluated."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=5,
            noise_types=["headers_footers"]
        )
        
        clean_result = results["clean"]
        assert clean_result is not None
        assert clean_result.metrics is not None
        assert clean_result.metrics.total_claims == 5
    
    def test_all_noise_types_supported(self, runner):
        """Test that all specified noise types are evaluated."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=5,
            noise_types=["headers_footers", "ocr_typos", "column_shuffle", "all"]
        )
        
        assert len(results) == 5, "Should have clean + 4 noise variants"
        assert "clean" in results
        assert "headers_footers" in results
        assert "ocr_typos" in results
        assert "column_shuffle" in results
        assert "all" in results
    
    def test_degradation_metrics_computed(self, runner):
        """Test that degradation metrics are computed."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=5,
            noise_types=["ocr_typos"]
        )
        
        noisy_metrics = results["ocr_typos"].metrics
        
        # Should have ingestion_noise_results
        assert noisy_metrics.ingestion_noise_results is not None
        assert "ocr_typos" in noisy_metrics.ingestion_noise_results
        
        degradation = noisy_metrics.ingestion_noise_results["ocr_typos"]
        assert "accuracy_drop" in degradation
        assert "clean_accuracy" in degradation
        assert "noisy_accuracy" in degradation
    
    def test_accuracy_degradation_reasonable(self, runner):
        """Test that accuracy degradation is in reasonable range."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=10,
            noise_types=["headers_footers"]
        )
        
        clean_acc = results["clean"].metrics.accuracy
        noisy_acc = results["headers_footers"].metrics.accuracy
        
        # Accuracy should still be between 0 and 1
        assert 0.0 <= clean_acc <= 1.0, "Clean accuracy should be valid"
        assert 0.0 <= noisy_acc <= 1.0, "Noisy accuracy should be valid"
        
        # Degradation should be reasonable (not catastrophic on small sample)
        degradation = clean_acc - noisy_acc
        assert abs(degradation) <= 1.0, "Degradation should be at most 100%"
    
    def test_predictions_count_matches(self, runner):
        """Test that prediction counts match for clean and noisy."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=8,
            noise_types=["ocr_typos"]
        )
        
        clean_count = len(results["clean"].predictions)
        noisy_count = len(results["ocr_typos"].predictions)
        
        assert clean_count == 8, "Should have 8 clean predictions"
        assert noisy_count == 8, "Should have 8 noisy predictions"
        assert clean_count == noisy_count, "Counts should match"
    
    def test_multiple_noise_types_independent(self, runner):
        """Test that multiple noise types are evaluated independently."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=5,
            noise_types=["headers_footers", "ocr_typos"]
        )
        
        # Each should have its own metrics
        headers_metrics = results["headers_footers"].metrics
        ocr_metrics = results["ocr_typos"].metrics
        
        # They may have different accuracy (noise affects differently)
        # Just verify both are valid
        assert 0.0 <= headers_metrics.accuracy <= 1.0
        assert 0.0 <= ocr_metrics.accuracy <= 1.0
    
    def test_column_shuffle_preserves_content(self, runner):
        """Test that column shuffle doesn't break evaluation."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=5,
            noise_types=["column_shuffle"]
        )
        
        shuffle_result = results["column_shuffle"]
        assert shuffle_result is not None
        assert shuffle_result.metrics.total_claims == 5
        assert len(shuffle_result.predictions) == 5
    
    def test_all_noise_combines_effects(self, runner):
        """Test that 'all' noise type applies multiple transformations."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=5,
            noise_types=["all"]
        )
        
        all_noise_result = results["all"]
        assert all_noise_result is not None
        
        # 'all' noise typically causes more degradation than individual types
        # Just verify it completes and produces valid metrics
        assert 0.0 <= all_noise_result.metrics.accuracy <= 1.0
    
    def test_calibration_metrics_tracked(self, runner):
        """Test that calibration metrics are tracked for noisy variants."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=10,
            noise_types=["headers_footers"]
        )
        
        clean_ece = results["clean"].metrics.ece
        noisy_ece = results["headers_footers"].metrics.ece
        
        # ECE should be valid (0 to 1)
        assert 0.0 <= clean_ece <= 1.0, "Clean ECE should be valid"
        assert 0.0 <= noisy_ece <= 1.0, "Noisy ECE should be valid"
        
        # Check if ECE increase is tracked
        degradation = results["headers_footers"].metrics.ingestion_noise_results["headers_footers"]
        assert "ece_increase" in degradation
    
    def test_evidence_coverage_tracked(self, runner):
        """Test that evidence coverage is tracked for robustness."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=8,
            noise_types=["ocr_typos"]
        )
        
        clean_coverage = results["clean"].metrics.evidence_coverage_rate
        noisy_coverage = results["ocr_typos"].metrics.evidence_coverage_rate
        
        # Coverage should be between 0 and 1
        assert 0.0 <= clean_coverage <= 1.0
        assert 0.0 <= noisy_coverage <= 1.0
        
        # Check if coverage drop is tracked
        degradation = results["ocr_typos"].metrics.ingestion_noise_results["ocr_typos"]
        assert "evidence_coverage_drop" in degradation
    
    def test_timing_metrics_reasonable(self, runner):
        """Test that timing is tracked and reasonable."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=5,
            noise_types=["headers_footers"]
        )
        
        # Both clean and noisy should have timing metrics
        clean_time = results["clean"].metrics.avg_time_per_claim
        noisy_time = results["headers_footers"].metrics.avg_time_per_claim
        
        assert clean_time > 0, "Clean timing should be positive"
        assert noisy_time > 0, "Noisy timing should be positive"
        
        # Timing should be reasonable (< 10 seconds per claim on CPU)
        assert clean_time < 10.0, "Clean time should be reasonable"
        assert noisy_time < 10.0, "Noisy time should be reasonable"
    
    def test_f1_scores_tracked(self, runner):
        """Test that per-label F1 scores are tracked."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=10,
            noise_types=["ocr_typos"]
        )
        
        degradation = results["ocr_typos"].metrics.ingestion_noise_results["ocr_typos"]
        
        # F1 drops should be present
        assert "f1_entail_drop" in degradation
        assert "f1_contradict_drop" in degradation
        
        # Drops should be reasonable
        assert abs(degradation["f1_entail_drop"]) <= 1.0
        assert abs(degradation["f1_contradict_drop"]) <= 1.0


class TestRobustnessReportGeneration:
    """Test robustness evaluation report generation."""
    
    @pytest.fixture
    def benchmark_dataset_path(self):
        """Path to CSClaimBench v1.0 dataset."""
        return Path(__file__).parent.parent / "evaluation" / "cs_benchmark" / "csclaimbench_v1.jsonl"
    
    @pytest.fixture
    def runner(self, benchmark_dataset_path):
        """Create benchmark runner."""
        return CSBenchmarkRunner(
            dataset_path=str(benchmark_dataset_path),
            batch_size=4,
            device="cpu",
            seed=42,
            log_predictions=False
        )
    
    def test_results_can_be_saved(self, runner, tmp_path):
        """Test that robustness results can be saved to disk."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=5,
            noise_types=["headers_footers"]
        )
        
        # Save clean result
        clean_json_path = tmp_path / "clean_metrics.json"
        results["clean"].to_json(clean_json_path)
        assert clean_json_path.exists()
        
        # Save noisy result
        noisy_json_path = tmp_path / "headers_footers_metrics.json"
        results["headers_footers"].to_json(noisy_json_path)
        assert noisy_json_path.exists()
    
    def test_degradation_summary_accessible(self, runner):
        """Test that degradation summary is easily accessible."""
        results = runner.run_ingestion_robustness_eval(
            sample_size=5,
            noise_types=["ocr_typos", "headers_footers"]
        )
        
        # Create summary of degradation
        summary = {}
        for noise_type, result in results.items():
            if noise_type == "clean":
                continue
            
            degradation = result.metrics.ingestion_noise_results.get(noise_type)
            if degradation:
                summary[noise_type] = {
                    "accuracy_drop": degradation["accuracy_drop"],
                    "ece_increase": degradation["ece_increase"]
                }
        
        assert len(summary) == 2, "Should have 2 noise types in summary"
        assert "ocr_typos" in summary
        assert "headers_footers" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
