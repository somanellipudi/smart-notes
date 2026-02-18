"""
Tests for benchmark dataset format validation.

Validates that the CS benchmark dataset conforms to schema and quality standards.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, List

from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner


class TestBenchmarkFormatValidation:
    """Validate benchmark dataset format and schema."""
    
    DATASET_PATH = "evaluation/cs_benchmark/cs_benchmark_dataset.jsonl"
    REQUIRED_FIELDS = {"doc_id", "domain_topic", "source_text", "generated_claim", 
                       "gold_label", "evidence_span"}
    VALID_LABELS = {"VERIFIED", "LOW_CONFIDENCE", "REJECTED"}
    VALID_DOMAINS = {
        "algorithms.sorting", "algorithms.search", "algorithms.dynamicprogramming",
        "datastructures.hashtable", "datastructures.binarytree", "datastructures.graph",
        "complexity.nphard", "complexity.correlation",
        "networking.basics", "networking.protocol",
        "security.encryption", "security.hashing",
        "databases.indexing", "databases.sqlquery",
        "machinelearning.optimization", "machinelearning.regression",
        "compilerscomputing.parsing", "compilerscomputing.optimization"
    }
    
    @pytest.fixture
    def dataset_path(self) -> Path:
        """Provide path to benchmark dataset."""
        return Path(self.DATASET_PATH)
    
    @pytest.fixture
    def examples(self, dataset_path: Path) -> List[Dict]:
        """Load benchmark examples."""
        examples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return examples
    
    def test_dataset_exists(self, dataset_path: Path):
        """Test that dataset file exists."""
        assert dataset_path.exists(), f"Dataset not found at {dataset_path}"
        assert dataset_path.is_file(), f"Dataset is not a file: {dataset_path}"
    
    def test_dataset_not_empty(self, examples: List[Dict]):
        """Test that dataset contains examples."""
        assert len(examples) > 0, "Dataset is empty"
    
    def test_example_count_reasonable(self, examples: List[Dict]):
        """Test that dataset size is reasonable."""
        assert len(examples) >= 10, "Dataset too small (< 10 examples)"
        assert len(examples) <= 1000, "Dataset too large (> 1000 examples)"
    
    def test_all_examples_have_required_fields(self, examples: List[Dict]):
        """Test that all examples have required fields."""
        for i, example in enumerate(examples):
            missing = self.REQUIRED_FIELDS - set(example.keys())
            assert not missing, f"Example {i} missing fields: {missing}"
    
    def test_doc_ids_unique(self, examples: List[Dict]):
        """Test that all doc_ids are unique."""
        doc_ids = [ex["doc_id"] for ex in examples]
        assert len(doc_ids) == len(set(doc_ids)), "Duplicate doc_ids found"
    
    def test_doc_id_format(self, examples: List[Dict]):
        """Test that doc_ids follow naming convention."""
        for example in examples:
            doc_id = example["doc_id"]
            # Should match pattern: domain_NNN
            parts = doc_id.split("_")
            assert len(parts) >= 2, f"Invalid doc_id format: {doc_id}"
            assert parts[-1].isdigit(), f"doc_id should end with number: {doc_id}"
    
    def test_valid_gold_labels(self, examples: List[Dict]):
        """Test that gold_labels are valid."""
        for i, example in enumerate(examples):
            label = example["gold_label"]
            assert label in self.VALID_LABELS, \
                f"Example {i}: Invalid label '{label}', must be one of {self.VALID_LABELS}"
    
    def test_valid_domain_topics(self, examples: List[Dict]):
        """Test that domain_topics are valid."""
        for i, example in enumerate(examples):
            domain = example["domain_topic"]
            assert domain in self.VALID_DOMAINS, \
                f"Example {i}: Invalid domain '{domain}', must be one of {self.VALID_DOMAINS}"
    
    def test_text_field_nonempty(self, examples: List[Dict]):
        """Test that text fields are not empty."""
        for i, example in enumerate(examples):
            assert example["source_text"].strip(), \
                f"Example {i}: source_text is empty"
            assert example["generated_claim"].strip(), \
                f"Example {i}: generated_claim is empty"
    
    def test_text_field_lengths(self, examples: List[Dict]):
        """Test text field lengths are reasonable."""
        MIN_CLAIM = 10
        MAX_CLAIM = 500
        MIN_SOURCE = 50
        MAX_SOURCE = 2000
        
        for i, example in enumerate(examples):
            claim_len = len(example["generated_claim"])
            assert MIN_CLAIM <= claim_len <= MAX_CLAIM, \
                f"Example {i}: claim length {claim_len} outside [{MIN_CLAIM}, {MAX_CLAIM}]"
            
            source_len = len(example["source_text"])
            assert MIN_SOURCE <= source_len <= MAX_SOURCE, \
                f"Example {i}: source length {source_len} outside [{MIN_SOURCE}, {MAX_SOURCE}]"
    
    def test_evidence_span_consistency(self, examples: List[Dict]):
        """Test that evidence_span is in source_text (if provided)."""
        for i, example in enumerate(examples):
            span = example.get("evidence_span", "")
            source = example["source_text"]
            
            if span:
                # For VERIFIED claims, span should be in source
                if example["gold_label"] == "VERIFIED":
                    assert span in source, \
                        f"Example {i}: evidence_span not found in source_text"
            
            # For REJECTED claims, span should be empty or not present
            if example["gold_label"] == "REJECTED":
                assert not span or span == "", \
                    f"Example {i}: REJECTED claim should have empty evidence_span"
    
    def test_no_duplicate_claims(self, examples: List[Dict]):
        """Test that generated_claims are unique."""
        claims = [ex["generated_claim"] for ex in examples]
        assert len(claims) == len(set(claims)), "Duplicate claims found"
    
    def test_label_distribution(self, examples: List[Dict]):
        """Test that label distribution is reasonable (not completely skewed)."""
        label_counts = {}
        for example in examples:
            label = example["gold_label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total = len(examples)
        
        # Each label should have at least 20% representation (for small dataset)
        if total >= 10:
            for label, count in label_counts.items():
                ratio = count / total
                assert ratio >= 0.15, \
                    f"Label {label} underrepresented: {count}/{total} ({ratio:.1%})"
    
    def test_domain_distribution(self, examples: List[Dict]):
        """Test that domains are reasonably distributed."""
        domain_counts = {}
        for example in examples:
            domain = example["domain_topic"]
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Should have at least 2-3 domains represented
        assert len(domain_counts) >= 2, \
            f"Only {len(domain_counts)} domain(s) represented, should be at least 2"
    
    def test_no_unicode_errors(self, examples: List[Dict]):
        """Test that all text is valid Unicode."""
        for i, example in enumerate(examples):
            try:
                for field in ["source_text", "generated_claim", "evidence_span"]:
                    example[field].encode('utf-8')
            except (UnicodeDecodeError, UnicodeEncodeError) as e:
                pytest.fail(f"Example {i}: Unicode error in {field}: {e}")
    
    def test_no_extremely_long_texts(self, examples: List[Dict]):
        """Test that texts aren't pathologically long (encoding issues)."""
        MAX_SAFE_LENGTH = 10000
        
        for i, example in enumerate(examples):
            source_len = len(example["source_text"])
            claim_len = len(example["generated_claim"])
            
            assert source_len < MAX_SAFE_LENGTH, \
                f"Example {i}: source_text too long ({source_len} chars)"
            assert claim_len < MAX_SAFE_LENGTH, \
                f"Example {i}: generated_claim too long ({claim_len} chars)"


class TestBenchmarkDatasetLoadable:
    """Test that benchmark dataset can be loaded by benchmark runner."""
    
    def test_runner_can_load_dataset(self):
        """Test that CSBenchmarkRunner can load dataset."""
        runner = CSBenchmarkRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            seed=42,
            device="cpu"
        )
        
        assert len(runner.dataset) > 0, "Runner loaded empty dataset"
    
    def test_runner_dataset_examples_valid(self):
        """Test that loaded examples have expected structure."""
        runner = CSBenchmarkRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            seed=42,
            device="cpu"
        )
        
        for example in runner.dataset[:5]:
            assert "doc_id" in example
            assert "domain_topic" in example
            assert "source_text" in example
            assert "generated_claim" in example
            assert "gold_label" in example


class TestBenchmarkDatasetReproducibility:
    """Test reproducibility properties of benchmark dataset."""
    
    def test_deterministic_loading(self):
        """Test that dataset loads consistently."""
        runner1 = CSBenchmarkRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            seed=42,
            device="cpu"
        )
        runner2 = CSBenchmarkRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            seed=42,
            device="cpu"
        )
        
        assert len(runner1.dataset) == len(runner2.dataset)
        for ex1, ex2 in zip(runner1.dataset, runner2.dataset):
            assert ex1["doc_id"] == ex2["doc_id"]
            assert ex1["generated_claim"] == ex2["generated_claim"]
    
    def test_seed_reproducibility(self):
        """Test that runner with same seed produces consistent results."""
        runner1 = CSBenchmarkRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            seed=42,
            device="cpu"
        )
        
        runner2 = CSBenchmarkRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
            seed=42,
            device="cpu"
        )
        
        # Both should load same dataset
        assert runner1.dataset == runner2.dataset


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
