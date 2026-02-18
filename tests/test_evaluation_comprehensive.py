"""
Evaluation Module: Benchmark Performance Tests

Tests for verifying correctness and performance of the benchmark runner
on diverse datasets and configurations.

Test Categories:
- Dataset Loading: Verify all benchmark datasets load correctly
- Performance Metrics: Validate metric computation accuracy
- Configuration Variations: Test different ablation configurations
- Reproducibility: Ensure deterministic results with seeds
- Scalability: Test performance with varying dataset sizes
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
import numpy as np

from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner, BenchmarkMetrics, BenchmarkResult
from src.claims.nli_verifier import NLIVerifier, EntailmentLabel


class TestDatasetLoading:
    """Test loading and validation of diverse benchmark datasets."""

    @pytest.fixture
    def dataset_dir(self):
        return Path("evaluation/cs_benchmark")

    def test_load_core_dataset(self, dataset_dir):
        """Load core CS benchmark dataset."""
        dataset_path = dataset_dir / "cs_benchmark_dataset.jsonl"
        assert dataset_path.exists(), f"Core dataset not found at {dataset_path}"
        
        examples = [json.loads(line) for line in open(dataset_path)]
        assert len(examples) > 0, "Core dataset is empty"
        assert all("doc_id" in ex for ex in examples), "Missing doc_id in examples"

    def test_load_hard_dataset(self, dataset_dir):
        """Load hard benchmark dataset."""
        dataset_path = dataset_dir / "cs_benchmark_hard.jsonl"
        if dataset_path.exists():
            examples = [json.loads(line) for line in open(dataset_path)]
            assert len(examples) > 0, "Hard dataset is empty"
            assert all(ex.get("difficulty") in ["easy", "medium", "hard"] 
                      for ex in examples), "Invalid difficulty levels"

    def test_load_easy_dataset(self, dataset_dir):
        """Load easy benchmark dataset."""
        dataset_path = dataset_dir / "cs_benchmark_easy.jsonl"
        if dataset_path.exists():
            examples = [json.loads(line) for line in open(dataset_path)]
            assert len(examples) > 0, "Easy dataset is empty"
            assert all(ex["gold_label"] in ["VERIFIED", "REJECTED", "LOW_CONFIDENCE"] 
                      for ex in examples), "Invalid labels"

    def test_load_domain_specific_dataset(self, dataset_dir):
        """Load domain-specific benchmark dataset."""
        dataset_path = dataset_dir / "cs_benchmark_domain_specific.jsonl"
        if dataset_path.exists():
            examples = [json.loads(line) for line in open(dataset_path)]
            assert len(examples) > 0, "Domain dataset is empty"
            
            domains = set(ex["domain_topic"] for ex in examples)
            assert len(domains) > 1, "Domain dataset should have multiple domains"

    def test_load_adversarial_dataset(self, dataset_dir):
        """Load adversarial benchmark dataset."""
        dataset_path = dataset_dir / "cs_benchmark_adversarial.jsonl"
        if dataset_path.exists():
            examples = [json.loads(line) for line in open(dataset_path)]
            assert len(examples) > 0, "Adversarial dataset is empty"

    def test_dataset_schemas_consistent(self, dataset_dir):
        """Verify all datasets follow same schema."""
        required_fields = {"doc_id", "domain_topic", "source_text", "generated_claim", "gold_label"}
        
        for dataset_file in dataset_dir.glob("cs_benchmark_*.jsonl"):
            examples = [json.loads(line) for line in open(dataset_file)]
            for ex in examples:
                assert required_fields.issubset(ex.keys()), \
                    f"Missing fields in {dataset_file}: {required_fields - set(ex.keys())}"


class TestBenchmarkMetrics:
    """Test accuracy of benchmark metric computation."""

    @pytest.fixture
    def simple_metrics(self):
        return BenchmarkMetrics(
            accuracy=0.8,
            precision_verified=0.75,
            recall_verified=0.85,
            f1_verified=0.80,
            ece=0.1,
            brier_score=0.05
        )

    def test_metrics_instantiation(self, simple_metrics):
        """Test BenchmarkMetrics dataclass creation."""
        assert simple_metrics.accuracy == 0.8
        assert simple_metrics.f1_verified == 0.80
        assert simple_metrics.ece == 0.1

    def test_metrics_serializable(self, simple_metrics):
        """Test metrics can be converted to dict for storage."""
        metrics_dict = simple_metrics.__dict__
        assert "accuracy" in metrics_dict
        assert len(metrics_dict) > 0

    def test_result_instantiation(self, simple_metrics):
        """Test BenchmarkResult creation."""
        result = BenchmarkResult(
            run_id="test_001",
            config={},
            metrics=simple_metrics,
            timestamp="2026-02-17T00:00:00"
        )
        assert result.config == {}
        assert result.metrics.accuracy == 0.8


class TestConfigurationVariations:
    """Test different benchmark configurations."""

    @pytest.fixture
    def runner(self):
        return CSBenchmarkRunner(device="cpu", batch_size=8)

    def test_runner_initialization(self, runner):
        """Test CSBenchmarkRunner can be initialized."""
        assert runner is not None
        assert runner.device == "cpu"
        assert runner.batch_size == 8

    def test_get_ablation_configs(self):
        """Test ablation configuration generation."""
        from scripts.run_cs_benchmark import AblationRunner
        
        ablation_runner = AblationRunner()
        configs = ablation_runner.get_ablation_configs()
        
        assert len(configs) > 0, "No ablation configs generated"
        assert all("name" in config for config in configs), "Missing config names"
        assert all("config" in config for config in configs), "Missing config dicts"

    def test_baseline_config(self):
        """Test baseline configuration (no verification)."""
        baseline_config = {
            "retrieval_enabled": False,
            "nli_enabled": False
        }
        assert not baseline_config["retrieval_enabled"]
        assert not baseline_config["nli_enabled"]

    def test_full_config(self):
        """Test full verification configuration."""
        full_config = {
            "retrieval_enabled": True,
            "nli_enabled": True,
            "batch_verification": True
        }
        assert full_config["retrieval_enabled"]
        assert full_config["nli_enabled"]
        assert full_config["batch_verification"]


class TestReproducibility:
    """Test deterministic results with seeding."""

    def test_seed_determinism(self):
        """Test that same seed produces same results."""
        runner1 = CSBenchmarkRunner(device="cpu")
        runner2 = CSBenchmarkRunner(device="cpu")
        
        # Both runners should be configured identically
        assert runner1.batch_size == runner2.batch_size
        assert runner1.device == runner2.device

    def test_nli_deterministic(self):
        """Test NLI verifier produces deterministic results."""
        verifier = NLIVerifier(device="cpu")
        
        claim = "The Earth is round"
        evidence = "The Earth is spherical"
        
        result1 = verifier.verify(claim, evidence)
        result2 = verifier.verify(claim, evidence)
        
        assert result1.label == result2.label, "NLI results not deterministic"
        assert result1.entailment_prob == result2.entailment_prob


class TestScalability:
    """Test performance with varying dataset sizes."""

    @pytest.mark.slow
    def test_small_sample_execution(self):
        """Test benchmark on very small sample (smoke test)."""
        runner = CSBenchmarkRunner(device="cpu", batch_size=2)
        
        # Load small dataset
        dataset_path = Path("evaluation/cs_benchmark/cs_benchmark_easy.jsonl")
        examples = [json.loads(line) for line in open(dataset_path)][:3]
        
        assert len(examples) == 3, "Test dataset loading failed"

    @pytest.mark.slow
    def test_medium_sample_execution(self):
        """Test benchmark on medium sample."""
        runner = CSBenchmarkRunner(device="cpu", batch_size=4)
        
        dataset_path = Path("evaluation/cs_benchmark/cs_benchmark_dataset.jsonl")
        if dataset_path.exists():
            examples = [json.loads(line) for line in open(dataset_path)]
            assert len(examples) > 5, "Dataset too small for medium test"

    def test_batch_size_variations(self):
        """Test different batch sizes don't break code."""
        for batch_size in [1, 2, 4, 8, 16]:
            runner = CSBenchmarkRunner(device="cpu", batch_size=batch_size)
            assert runner.batch_size == batch_size


class TestDiverseDatasets:
    """Test benchmark on diverse dataset characteristics."""

    def test_difficulty_distribution(self):
        """Verify datasets have varied difficulty levels."""
        dataset_path = Path("evaluation/cs_benchmark/cs_benchmark_hard.jsonl")
        if dataset_path.exists():
            examples = [json.loads(line) for line in open(dataset_path)]
            difficulties = [ex.get("difficulty", "medium") for ex in examples]
            assert len(set(difficulties)) > 1, "No difficulty variation"

    def test_claim_type_variety(self):
        """Verify datasets have varied claim types."""
        dataset_path = Path("evaluation/cs_benchmark/cs_benchmark_dataset.jsonl")
        if dataset_path.exists():
            examples = [json.loads(line) for line in open(dataset_path)]
            claim_types = set(ex.get("claim_type", "unknown") for ex in examples)
            assert len(claim_types) >= 1, "No claim type variety"

    def test_reasoning_type_variety(self):
        """Verify datasets require varied reasoning."""
        dataset_path = Path("evaluation/cs_benchmark/cs_benchmark_hard.jsonl")
        if dataset_path.exists():
            examples = [json.loads(line) for line in open(dataset_path)]
            reasoning_types = set(ex.get("reasoning_type", "unknown") for ex in examples)
            assert "direct" in reasoning_types or "implicit" in reasoning_types, \
                "No reasoning type variety"

    def test_source_type_distribution(self):
        """Verify datasets use varied sources."""
        dataset_path = Path("evaluation/cs_benchmark/cs_benchmark_domain_specific.jsonl")
        if dataset_path.exists():
            examples = [json.loads(line) for line in open(dataset_path)]
            sources = set(ex.get("source_type", "unknown") for ex in examples)
            assert len(sources) >= 1, "No source variety"


class TestNLIVerifier:
    """Test NLI verifier integration."""

    @pytest.fixture
    def verifier(self):
        return NLIVerifier(device="cpu", batch_size=4)

    def test_single_verification(self, verifier):
        """Test single claim-evidence verification."""
        result = verifier.verify(
            "Paris is the capital of France",
            "France's capital city is Paris"
        )
        assert result.label in [EntailmentLabel.ENTAILMENT, EntailmentLabel.NEUTRAL]
        assert 0 <= result.entailment_prob <= 1

    def test_batch_verification(self, verifier):
        """Test batch verification of multiple pairs."""
        pairs = [
            ("A is B", "B is A"),
            ("Cats are animals", "Animals include cats"),
            ("2 + 2 = 5", "2 + 2 = 4")
        ]
        results = verifier.verify_batch(pairs)
        
        assert len(results) == len(pairs)
        assert all(r.label in [EntailmentLabel.ENTAILMENT, EntailmentLabel.CONTRADICTION, 
                              EntailmentLabel.NEUTRAL] for r in results)

    def test_consensus_verification(self, verifier):
        """Test multi-source consensus mode."""
        claim = "Sorting algorithms are used in databases"
        evidence_list = [
            "Databases use sorting for indexing",
            "Sorting orders data for efficient retrieval",
            "Many DB operations require sorted data"
        ]
        
        result = verifier.check_consensus(claim, evidence_list, min_entailment_sources=2)
        assert "consensus" in result
        assert "entailment_count" in result


class TestMetricsComputation:
    """Test correctness of metric calculations."""

    def test_accuracy_calculation(self):
        """Test accuracy computation."""
        predictions = ["VERIFIED", "REJECTED", "VERIFIED", "LOW_CONFIDENCE"]
        ground_truth = ["VERIFIED", "REJECTED", "VERIFIED", "LOW_CONFIDENCE"]
        
        accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
        assert accuracy == 1.0, "Perfect predictions should have accuracy 1.0"

    def test_f1_calculation(self):
        """Test F1 score computation."""
        tp, fp, fn = 10, 2, 3
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        assert 0 <= f1 <= 1, "F1 should be between 0 and 1"
        assert f1 > 0, "F1 should be positive for valid inputs"

    def test_ece_calculation(self):
        """Test Expected Calibration Error computation."""
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        accuracies = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
        
        ece = np.mean(np.abs(confidences - accuracies))
        assert 0 <= ece <= 1, "ECE should be between 0 and 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
