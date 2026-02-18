"""
Smoke tests for ablation runner.

Quick tests to ensure ablation infrastructure works (CI-friendly).
Runs on small sample size for speed.
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd

from scripts.run_cs_benchmark import AblationRunner


class TestAblationRunnerSmoke:
    """Smoke tests for AblationRunner (quick, small sample)."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Provide temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_runner_initialization(self, temp_output_dir):
        """Test that runner initializes without errors."""
        runner = AblationRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            output_dir=temp_output_dir,
            sample_size=5,
            seed=42
        )
        
        assert runner is not None
        assert runner.output_dir == Path(temp_output_dir)
        assert runner.sample_size == 5
    
    def test_runner_ablation_configs_available(self, temp_output_dir):
        """Test that ablation configurations are available."""
        runner = AblationRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            output_dir=temp_output_dir,
            sample_size=5
        )
        
        configs = runner.get_ablation_configs()
        
        assert len(configs) > 0
        assert "00_no_verification" in configs
        assert "01a_retrieval_only" in configs
        assert "01b_retrieval_nli" in configs
        assert "01c_ensemble" in configs
    
    def test_single_config_run(self, temp_output_dir):
        """Test running single configuration."""
        runner = AblationRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            output_dir=temp_output_dir,
            sample_size=3,  # Small sample for speed
            seed=42
        )
        
        # Get baseline config
        configs = runner.get_ablation_configs()
        config_name = "00_no_verification"
        config = configs[config_name]
        
        # Run benchmark
        from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner
        
        benchmark_runner = CSBenchmarkRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            seed=42,
            device="cpu"
        )
        
        result = benchmark_runner.run(
            config=config,
            sample_size=3
        )
        
        # Check result structure
        assert result is not None
        assert result.metrics.accuracy is not None
        assert 0 <= result.metrics.accuracy <= 1
    
    def test_ablations_complete_without_error(self, temp_output_dir):
        """Test that ablations run to completion (smoke test)."""
        runner = AblationRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            output_dir=temp_output_dir,
            sample_size=3,  # Small sample
            seed=42
        )
        
        # Run ablations (should not raise)
        try:
            df = runner.run_ablations(noise_injection=False)
            
            assert df is not None
            assert len(df) > 0
            
        except Exception as e:
            pytest.fail(f"Ablations failed with: {e}")
    
    def test_results_csv_created(self, temp_output_dir):
        """Test that results CSV is created."""
        runner = AblationRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            output_dir=temp_output_dir,
            sample_size=2,
            seed=42
        )
        
        runner.run_ablations(noise_injection=False)
        
        csv_path = Path(temp_output_dir) / "results.csv"
        assert csv_path.exists(), f"Results CSV not created at {csv_path}"
        
        # Verify CSV is readable
        df = pd.read_csv(csv_path)
        assert len(df) > 0
    
    def test_summary_markdown_created(self, temp_output_dir):
        """Test that summary markdown is created."""
        runner = AblationRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            output_dir=temp_output_dir,
            sample_size=2,
            seed=42
        )
        
        runner.run_ablations(noise_injection=False)
        
        summary_path = Path(temp_output_dir) / "ablation_summary.md"
        assert summary_path.exists(), f"Summary markdown not created at {summary_path}"
        
        # Verify content
        with open(summary_path, 'r') as f:
            content = f.read()
            assert "Ablation Study Results" in content
            assert "Overall" in content or "Results" in content
    
    def test_detailed_results_created(self, temp_output_dir):
        """Test that detailed result files are created."""
        runner = AblationRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            output_dir=temp_output_dir,
            sample_size=2,
            seed=42
        )
        
        runner.run_ablations(noise_injection=False)
        
        detailed_dir = Path(temp_output_dir) / "detailed_results"
        assert detailed_dir.exists(), "detailed_results directory not created"
        
        # Should have some result files
        result_files = list(detailed_dir.glob("*.json"))
        assert len(result_files) > 0, "No result JSON files created"


class TestAblationRunnerOutputFormat:
    """Test output format and content of ablation runner."""
    
    @pytest.fixture
    def ablation_results(self):
        """Run quick ablation and provide results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = AblationRunner(
                dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
                output_dir=tmpdir,
                sample_size=3,
                seed=42
            )
            df = runner.run_ablations(noise_injection=False)
            yield df
    
    def test_results_dataframe_has_metrics(self, ablation_results):
        """Test that results dataframe has expected metrics columns."""
        assert "config_name" in ablation_results.columns
        assert "accuracy" in ablation_results.columns
        assert "F1_verified" in ablation_results.columns or "F1" in ablation_results.columns
    
    def test_results_metrics_in_valid_range(self, ablation_results):
        """Test that metrics are in valid ranges."""
        for _, row in ablation_results.iterrows():
            # Accuracy should be 0-1
            acc = row.get("accuracy")
            if pd.notna(acc):
                assert 0 <= acc <= 1, f"Accuracy out of range: {acc}"
    
    def test_results_have_all_configs(self, ablation_results):
        """Test that results include all expected configurations."""
        configs = ablation_results["config_name"].tolist()
        
        # Should have main ones
        expected_prefixes = ["00", "01a", "01b", "01c"]
        for prefix in expected_prefixes:
            found = any(c.startswith(prefix) for c in configs)
            assert found, f"No config starting with {prefix}"


class TestAblationRunnerReproducibility:
    """Test reproducibility of ablation runs."""
    
    def test_deterministic_with_seed(self):
        """Test that same seed produces consistent results."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                
                runner1 = AblationRunner(
                    dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
                    output_dir=tmpdir1,
                    sample_size=3,
                    seed=42
                )
                df1 = runner1.run_ablations(noise_injection=False)
                
                runner2 = AblationRunner(
                    dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
                    output_dir=tmpdir2,
                    sample_size=3,
                    seed=42
                )
                df2 = runner2.run_ablations(noise_injection=False)
                
                # Same configs should be tested
                assert sorted(df1["config_name"].tolist()) == sorted(df2["config_name"].tolist())


class TestAblationRunnerCLI:
    """Test CLI integration."""
    
    def test_cli_callable(self):
        """Test that CLI can be imported and called."""
        from scripts.run_cs_benchmark import main
        
        assert callable(main), "main() is not callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
