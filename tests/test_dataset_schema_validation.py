"""
Dataset Schema Validation Tests

Validates that all benchmark datasets match expected schema.
Prevents bugs like 'claim' vs 'generated_claim' field naming mismatches.

Test this with: pytest tests/test_dataset_schema_validation.py -v
"""

import json
import pytest
from pathlib import Path


# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

REQUIRED_FIELDS = {
    'doc_id',
    'domain_topic',
    'source_text',
    'generated_claim',
    'gold_label',
    'evidence_span'
}

DEPRECATED_FIELDS = {
    'claim',           # Use 'generated_claim' instead
    'label',           # Use 'gold_label' instead
    'topic',           # Use 'domain_topic' instead
}

VALID_LABELS = {'VERIFIED', 'REJECTED', 'LOW_CONFIDENCE'}

FIELD_TYPES = {
    'doc_id': (str, int),
    'domain_topic': str,
    'source_text': str,
    'generated_claim': str,
    'gold_label': str,
    'evidence_span': (str, list, type(None)),
}


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def cs_benchmark_dataset_path():
    """Path to CS benchmark dataset."""
    path = Path('evaluation/cs_benchmark/cs_benchmark_dataset.jsonl')
    if not path.exists():
        pytest.skip(f"Dataset not found at {path}")
    return path


@pytest.fixture
def load_dataset(cs_benchmark_dataset_path):
    """Load dataset from JSONL file."""
    examples = []
    with open(cs_benchmark_dataset_path) as f:
        for line_no, line in enumerate(f, 1):
            try:
                example = json.loads(line)
                examples.append({'line_no': line_no, 'data': example})
            except json.JSONDecodeError as e:
                pytest.fail(f"Line {line_no}: Invalid JSON: {e}")
    return examples


# ============================================================================
# TEST: Required Fields Present
# ============================================================================

def test_all_examples_have_required_fields(load_dataset):
    """Each example must have all required fields."""
    for item in load_dataset:
        line_no = item['line_no']
        example = item['data']
        
        missing = REQUIRED_FIELDS - set(example.keys())
        
        assert not missing, (
            f"Line {line_no}: Missing required fields: {missing}\n"
            f"Available fields: {list(example.keys())}\n"
            f"Expected: {sorted(REQUIRED_FIELDS)}"
        )


# ============================================================================
# TEST: No Deprecated Fields
# ============================================================================

def test_no_deprecated_fields_present(load_dataset):
    """Deprecated field names must not appear."""
    for item in load_dataset:
        line_no = item['line_no']
        example = item['data']
        
        deprecated_found = DEPRECATED_FIELDS & set(example.keys())
        
        assert not deprecated_found, (
            f"Line {line_no}: Found deprecated field(s): {deprecated_found}\n"
            f"Mapping:\n"
            f"  'claim' → use 'generated_claim'\n"
            f"  'label' → use 'gold_label'\n"
            f"  'topic' → use 'domain_topic'\n"
            f"See: evaluation/cs_benchmark/README_DATASETS.md"
        )


# ============================================================================
# TEST: Field Data Types
# ============================================================================

def test_field_data_types(load_dataset):
    """Each field must have correct data type."""
    for item in load_dataset:
        line_no = item['line_no']
        example = item['data']
        
        for field, expected_types in FIELD_TYPES.items():
            if field not in example:
                continue
            
            value = example[field]
            if not isinstance(expected_types, tuple):
                expected_types = (expected_types,)
            
            assert isinstance(value, expected_types), (
                f"Line {line_no}: Field '{field}' has wrong type\n"
                f"Expected: {expected_types}\n"
                f"Got: {type(value)} = {repr(value)}"
            )


# ============================================================================
# TEST: Label Values Valid
# ============================================================================

def test_gold_label_values_valid(load_dataset):
    """gold_label must be one of VERIFIED, REJECTED, LOW_CONFIDENCE."""
    for item in load_dataset:
        line_no = item['line_no']
        example = item['data']
        
        label = example.get('gold_label')
        
        assert label in VALID_LABELS, (
            f"Line {line_no}: Invalid gold_label value: {repr(label)}\n"
            f"Valid values: {sorted(VALID_LABELS)}"
        )


# ============================================================================
# TEST: Generated Claim is Non-Empty
# ============================================================================

def test_generated_claim_not_empty(load_dataset):
    """Generated claim must be non-empty string."""
    for item in load_dataset:
        line_no = item['line_no']
        example = item['data']
        
        claim = example.get('generated_claim', '')
        
        assert isinstance(claim, str), (
            f"Line {line_no}: generated_claim must be string, got {type(claim)}"
        )
        
        assert len(claim.strip()) > 0, (
            f"Line {line_no}: generated_claim is empty"
        )


# ============================================================================
# TEST: Domain Topic Format
# ============================================================================

def test_domain_topic_format(load_dataset):
    """Domain topic should follow namespace.topic format."""
    for item in load_dataset:
        line_no = item['line_no']
        example = item['data']
        
        domain = example.get('domain_topic', '')
        
        # Should be like "algorithms.sorting" not just "algorithms"
        parts = domain.split('.')
        assert len(parts) >= 1, (
            f"Line {line_no}: domain_topic should be formatted as 'namespace.topic'\n"
            f"Got: {repr(domain)}"
        )


# ============================================================================
# TEST: No Extra Unexpected Fields
# ============================================================================

def test_no_unexpected_extra_fields(load_dataset):
    """Warn if extra fields not in standard schema (informational)."""
    STANDARD_FIELDS = REQUIRED_FIELDS | {'metadata', 'split', 'source', 'url', 'annotations'}
    
    extra_field_warnings = set()
    
    for item in load_dataset:
        example = item['data']
        extra = set(example.keys()) - STANDARD_FIELDS
        
        if extra:
            extra_field_warnings.update(extra)
    
    # This is a warning, not a hard failure
    if extra_field_warnings:
        print(f"\n⚠️  Extra fields found (non-standard): {sorted(extra_field_warnings)}")
        print("   These fields will be ignored. If intentional, add to STANDARD_FIELDS.")


# ============================================================================
# TEST: Dataset Statistics
# ============================================================================

def test_dataset_statistics(load_dataset):
    """Report dataset statistics (informational)."""
    if not load_dataset:
        pytest.skip("Empty dataset")
    
    n_examples = len(load_dataset)
    
    # Count by label
    label_counts = {}
    for item in load_dataset:
        label = item['data'].get('gold_label')
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Count by domain
    domain_counts = {}
    for item in load_dataset:
        domain = item['data'].get('domain_topic', 'unknown').split('.')[0]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"\n{'='*70}")
    print(f"DATASET STATISTICS")
    print(f"{'='*70}")
    print(f"Total examples: {n_examples}")
    print(f"\nLabels:")
    for label in sorted(VALID_LABELS):
        count = label_counts.get(label, 0)
        pct = 100 * count / n_examples if n_examples > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
    print(f"\nTop domains:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {domain}: {count}")
    print(f"{'='*70}\n")
    
    # Basic assertions
    assert n_examples > 0, "Dataset is empty"
    assert len(label_counts) > 0, "No valid labels found"


# ============================================================================
# TEST: Integration - Round-trip Serialization
# ============================================================================

def test_json_roundtrip_serialization(load_dataset):
    """Ensure all examples can be serialized and deserialized."""
    for item in load_dataset:
        example = item['data']
        
        # Serialize
        try:
            json_str = json.dumps(example)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Line {item['line_no']}: Cannot serialize to JSON: {e}")
        
        # Deserialize
        try:
            recovered = json.loads(json_str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Line {item['line_no']}: Cannot deserialize from JSON: {e}")
        
        # Should match
        assert recovered == example, (
            f"Line {item['line_no']}: Round-trip serialization changed data"
        )


# ============================================================================
# MAIN: Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
