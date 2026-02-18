"""
Test suite for CS Benchmark dataset format validation.

Validates JSONL schema, required fields, label values, evidence spans,
and domain coverage for the CSClaimBench v1.0 dataset.
"""

import json
import pytest
from pathlib import Path


# Path to benchmark dataset
BENCHMARK_PATH = Path(__file__).parent.parent / "evaluation" / "cs_benchmark" / "csclaimbench_v1.jsonl"

# Expected schema
REQUIRED_FIELDS = {"doc_id", "domain_topic", "source_text", "claim", "gold_label"}
OPTIONAL_FIELDS = {"evidence_span"}
VALID_LABELS = {"ENTAIL", "CONTRADICT", "NEUTRAL"}
VALID_DOMAINS = {"Algorithms", "DataStructures", "OS", "DB", "Distributed", "Networks", "Compilers"}


def load_benchmark_dataset():
    """Load the full benchmark dataset."""
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def dataset():
    """Fixture to load dataset once for all tests."""
    return load_benchmark_dataset()


class TestBenchmarkFormat:
    """Test JSONL format and schema validation."""
    
    def test_file_exists(self):
        """Verify benchmark file exists."""
        assert BENCHMARK_PATH.exists(), f"Benchmark file not found at {BENCHMARK_PATH}"
    
    def test_valid_jsonl(self, dataset):
        """Verify all lines are valid JSON."""
        assert len(dataset) > 0, "Dataset should not be empty"
        assert all(isinstance(item, dict) for item in dataset), "All lines must parse as JSON objects"
    
    def test_required_fields_present(self, dataset):
        """Verify all required fields are present in each example."""
        for idx, item in enumerate(dataset):
            missing = REQUIRED_FIELDS - set(item.keys())
            assert not missing, f"Example {idx} ({item.get('doc_id', 'unknown')}) missing fields: {missing}"
    
    def test_no_unexpected_fields(self, dataset):
        """Verify no unexpected fields are present."""
        allowed_fields = REQUIRED_FIELDS | OPTIONAL_FIELDS
        for idx, item in enumerate(dataset):
            unexpected = set(item.keys()) - allowed_fields
            assert not unexpected, f"Example {idx} ({item.get('doc_id', 'unknown')}) has unexpected fields: {unexpected}"
    
    def test_field_types(self, dataset):
        """Verify field types are correct."""
        for idx, item in enumerate(dataset):
            assert isinstance(item["doc_id"], str), f"Example {idx}: doc_id must be string"
            assert isinstance(item["domain_topic"], str), f"Example {idx}: domain_topic must be string"
            assert isinstance(item["source_text"], str), f"Example {idx}: source_text must be string"
            assert isinstance(item["claim"], str), f"Example {idx}: claim must be string"
            assert isinstance(item["gold_label"], str), f"Example {idx}: gold_label must be string"
            
            if "evidence_span" in item:
                assert isinstance(item["evidence_span"], dict), f"Example {idx}: evidence_span must be dict"
                assert "start" in item["evidence_span"], f"Example {idx}: evidence_span missing 'start'"
                assert "end" in item["evidence_span"], f"Example {idx}: evidence_span missing 'end'"
                assert isinstance(item["evidence_span"]["start"], int), f"Example {idx}: evidence_span.start must be int"
                assert isinstance(item["evidence_span"]["end"], int), f"Example {idx}: evidence_span.end must be int"


class TestLabelValidation:
    """Test gold labels are valid."""
    
    def test_valid_labels(self, dataset):
        """Verify all labels are one of ENTAIL/CONTRADICT/NEUTRAL."""
        for idx, item in enumerate(dataset):
            label = item["gold_label"]
            assert label in VALID_LABELS, \
                f"Example {idx} ({item['doc_id']}): invalid label '{label}', must be one of {VALID_LABELS}"
    
    def test_label_distribution(self, dataset):
        """Verify label distribution is reasonable."""
        label_counts = {label: 0 for label in VALID_LABELS}
        for item in dataset:
            label_counts[item["gold_label"]] += 1
        
        total = len(dataset)
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            print(f"{label}: {count} ({percentage:.1f}%)")
        
        # At least 10 examples of each label
        for label in VALID_LABELS:
            assert label_counts[label] >= 10, f"Label {label} has insufficient examples: {label_counts[label]}"


class TestDomainCoverage:
    """Test domain coverage and distribution."""
    
    def test_valid_domains(self, dataset):
        """Verify all domain_topic values are valid."""
        for idx, item in enumerate(dataset):
            domain = item["domain_topic"]
            assert domain in VALID_DOMAINS, \
                f"Example {idx} ({item['doc_id']}): invalid domain '{domain}', must be one of {VALID_DOMAINS}"
    
    def test_domain_distribution(self, dataset):
        """Verify domain distribution is reasonable."""
        domain_counts = {domain: 0 for domain in VALID_DOMAINS}
        for item in dataset:
            domain_counts[item["domain_topic"]] += 1
        
        print("\nDomain distribution:")
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain}: {count} examples")
        
        # Each domain should have at least 20 examples
        for domain in VALID_DOMAINS:
            assert domain_counts[domain] >= 20, f"Domain {domain} has insufficient examples: {domain_counts[domain]}"
    
    def test_doc_id_matches_domain(self, dataset):
        """Verify doc_id prefix matches domain_topic."""
        domain_prefixes = {
            "Algorithms": "algo_",
            "DataStructures": "ds_",
            "OS": "os_",
            "DB": "db_",
            "Distributed": "distributed_",
            "Networks": "networks_",
            "Compilers": "compilers_"
        }
        
        for idx, item in enumerate(dataset):
            domain = item["domain_topic"]
            doc_id = item["doc_id"]
            expected_prefix = domain_prefixes[domain]
            assert doc_id.startswith(expected_prefix), \
                f"Example {idx}: doc_id '{doc_id}' should start with '{expected_prefix}' for domain {domain}"


class TestEvidenceSpans:
    """Test evidence span validity."""
    
    def test_evidence_span_validity(self, dataset):
        """Verify evidence spans are valid character offsets."""
        for idx, item in enumerate(dataset):
            if "evidence_span" not in item:
                continue
            
            span = item["evidence_span"]
            start = span["start"]
            end = span["end"]
            source_text = item["source_text"]
            
            assert 0 <= start < len(source_text), \
                f"Example {idx} ({item['doc_id']}): evidence_span.start {start} out of range [0, {len(source_text)})"
            assert start < end <= len(source_text), \
                f"Example {idx} ({item['doc_id']}): evidence_span.end {end} must be > start {start} and <= {len(source_text)}"
            
            # Extract evidence text to verify it's meaningful
            evidence_text = source_text[start:end]
            assert len(evidence_text) > 10, \
                f"Example {idx} ({item['doc_id']}): evidence span too short ({len(evidence_text)} chars): '{evidence_text}'"
    
    def test_evidence_span_label_consistency(self, dataset):
        """Verify evidence spans are present for ENTAIL/CONTRADICT, optional for NEUTRAL."""
        entail_with_span = 0
        entail_without_span = 0
        contradict_with_span = 0
        contradict_without_span = 0
        neutral_with_span = 0
        
        for item in dataset:
            label = item["gold_label"]
            has_span = "evidence_span" in item
            
            if label == "ENTAIL":
                if has_span:
                    entail_with_span += 1
                else:
                    entail_without_span += 1
            elif label == "CONTRADICT":
                if has_span:
                    contradict_with_span += 1
                else:
                    contradict_without_span += 1
            elif label == "NEUTRAL" and has_span:
                neutral_with_span += 1
        
        print(f"\nEvidence span statistics:")
        print(f"  ENTAIL with span: {entail_with_span}, without: {entail_without_span}")
        print(f"  CONTRADICT with span: {contradict_with_span}, without: {contradict_without_span}")
        print(f"  NEUTRAL with span: {neutral_with_span}")
        
        # Most ENTAIL/CONTRADICT should have evidence spans
        total_entail = entail_with_span + entail_without_span
        total_contradict = contradict_with_span + contradict_without_span
        
        if total_entail > 0:
            entail_span_rate = entail_with_span / total_entail
            assert entail_span_rate >= 0.8, \
                f"Expected >= 80% ENTAIL examples to have evidence spans, got {entail_span_rate*100:.1f}%"
        
        if total_contradict > 0:
            contradict_span_rate = contradict_with_span / total_contradict
            assert contradict_span_rate >= 0.8, \
                f"Expected >= 80% CONTRADICT examples to have evidence spans, got {contradict_span_rate*100:.1f}%"


class TestContentQuality:
    """Test content quality and consistency."""
    
    def test_source_text_length(self, dataset):
        """Verify source texts have reasonable length."""
        for idx, item in enumerate(dataset):
            source_len = len(item["source_text"])
            assert source_len >= 50, \
                f"Example {idx} ({item['doc_id']}): source_text too short ({source_len} chars)"
            assert source_len <= 1000, \
                f"Example {idx} ({item['doc_id']}): source_text too long ({source_len} chars)"
    
    def test_claim_length(self, dataset):
        """Verify claims have reasonable length."""
        for idx, item in enumerate(dataset):
            claim_len = len(item["claim"])
            assert claim_len >= 10, \
                f"Example {idx} ({item['doc_id']}): claim too short ({claim_len} chars)"
            assert claim_len <= 200, \
                f"Example {idx} ({item['doc_id']}): claim too long ({claim_len} chars)"
    
    def test_unique_doc_ids(self, dataset):
        """Verify all doc_ids are unique."""
        doc_ids = [item["doc_id"] for item in dataset]
        duplicates = set([doc_id for doc_id in doc_ids if doc_ids.count(doc_id) > 1])
        assert not duplicates, f"Duplicate doc_ids found: {duplicates}"
    
    def test_no_empty_strings(self, dataset):
        """Verify no fields contain empty strings."""
        for idx, item in enumerate(dataset):
            assert item["doc_id"].strip(), f"Example {idx}: doc_id is empty"
            assert item["domain_topic"].strip(), f"Example {idx} ({item['doc_id']}): domain_topic is empty"
            assert item["source_text"].strip(), f"Example {idx} ({item['doc_id']}): source_text is empty"
            assert item["claim"].strip(), f"Example {idx} ({item['doc_id']}): claim is empty"
            assert item["gold_label"].strip(), f"Example {idx} ({item['doc_id']}): gold_label is empty"


class TestDatasetSize:
    """Test overall dataset size and completeness."""
    
    def test_minimum_size(self, dataset):
        """Verify dataset has at least 180 examples."""
        assert len(dataset) >= 180, f"Dataset should have >= 180 examples, found {len(dataset)}"
    
    def test_dataset_summary(self, dataset):
        """Print comprehensive dataset summary."""
        print(f"\n{'='*60}")
        print("CSClaimBench v1.0 Dataset Summary")
        print(f"{'='*60}")
        print(f"Total examples: {len(dataset)}")
        
        # Label distribution
        label_counts = {}
        for item in dataset:
            label = item["gold_label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\nLabel distribution:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = (count / len(dataset)) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        # Domain distribution
        domain_counts = {}
        for item in dataset:
            domain = item["domain_topic"]
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        print(f"\nDomain distribution:")
        for domain in sorted(domain_counts.keys()):
            count = domain_counts[domain]
            pct = (count / len(dataset)) * 100
            print(f"  {domain}: {count} ({pct:.1f}%)")
        
        # Evidence span coverage
        with_span = sum(1 for item in dataset if "evidence_span" in item)
        span_pct = (with_span / len(dataset)) * 100
        print(f"\nEvidence spans: {with_span}/{len(dataset)} ({span_pct:.1f}%)")
        print(f"{'='*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
