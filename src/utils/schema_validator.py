"""
Early schema validation utility for dataset access.

Provides clear error messages when dataset fields don't match expected schema.
Use this in dataset loading code to fail fast with helpful messages.

Example:
    from src.utils.schema_validator import validate_example_schema
    
    for example in dataset:
        validate_example_schema(example)  # Fails with clear error if schema wrong
"""

from typing import Dict, Any, Set


class SchemaValidationError(ValueError):
    """Raised when dataset example doesn't match expected schema."""
    pass


EXPECTED_SCHEMA = {
    'doc_id': (str, int),
    'domain_topic': str,
    'source_text': str,
    'generated_claim': str,
    'gold_label': str,
    'evidence_span': (str, list, type(None)),
}

REQUIRED_FIELDS: Set[str] = {
    'doc_id',
    'domain_topic',
    'source_text',
    'generated_claim',
    'gold_label',
    'evidence_span'
}

DEPRECATED_FIELDS = {
    'claim': 'Use "generated_claim" instead',
    'label': 'Use "gold_label" instead',
    'topic': 'Use "domain_topic" instead',
}

VALID_LABELS = {'VERIFIED', 'REJECTED', 'LOW_CONFIDENCE'}


def validate_example_schema(example: Dict[str, Any], example_id: str = "unknown") -> None:
    """
    Validate that dataset example matches expected schema.
    
    Provides clear, actionable error messages for:
    - Missing required fields
    - Deprecated field names
    - Invalid field types
    - Invalid label values
    
    Args:
        example: Dataset example dict
        example_id: Optional ID for error messages
    
    Raises:
        SchemaValidationError: If schema doesn't match
    
    Example:
        >>> example = {'generated_claim': 'X is true', ...}
        >>> validate_example_schema(example, example_id='doc_123')
        # Returns silently if valid
        
        >>> bad_example = {'claim': 'X is true'}  # Wrong field name
        >>> validate_example_schema(bad_example)
        SchemaValidationError: Deprecated field 'claim' found. Use 'generated_claim'
    """
    
    # Check for deprecated fields first (most common error)
    for deprecated_field, suggestion in DEPRECATED_FIELDS.items():
        if deprecated_field in example:
            raise SchemaValidationError(
                f"[{example_id}] Deprecated field '{deprecated_field}' found.\n"
                f"  {suggestion}\n"
                f"  See: evaluation/cs_benchmark/README_DATASETS.md"
            )
    
    # Check for missing required fields
    missing = REQUIRED_FIELDS - set(example.keys())
    if missing:
        raise SchemaValidationError(
            f"[{example_id}] Missing required fields: {sorted(missing)}\n"
            f"  Available fields: {sorted(example.keys())}\n"
            f"  Expected fields: {sorted(REQUIRED_FIELDS)}\n"
            f"  See: evaluation/cs_benchmark/README_DATASETS.md"
        )
    
    # Check field types
    for field, expected_type in EXPECTED_SCHEMA.items():
        if field not in example:
            continue
        
        value = example[field]
        
        if not isinstance(expected_type, tuple):
            expected_type = (expected_type,)
        
        if not isinstance(value, expected_type):
            raise SchemaValidationError(
                f"[{example_id}] Field '{field}' has wrong type.\n"
                f"  Expected: {expected_type}\n"
                f"  Got: {type(value).__name__} = {repr(value)}"
            )
    
    # Validate gold_label value
    gold_label = example.get('gold_label')
    if gold_label not in VALID_LABELS:
        raise SchemaValidationError(
            f"[{example_id}] Invalid gold_label: {repr(gold_label)}\n"
            f"  Valid values: {sorted(VALID_LABELS)}"
        )
    
    # Validate generated_claim is non-empty
    claim = example.get('generated_claim', '').strip()
    if not claim:
        raise SchemaValidationError(
            f"[{example_id}] generated_claim is empty or whitespace-only"
        )


def validate_dataset_batch(examples: list, max_errors: int = 5) -> Dict[str, Any]:
    """
    Validate a batch of examples and report any errors.
    
    Args:
        examples: List of dataset examples
        max_errors: Stop after finding this many errors (None for all)
    
    Returns:
        Dict with 'valid_count', 'invalid_count', 'errors'
    
    Example:
        >>> results = validate_dataset_batch(dataset[:100])
        >>> print(f"Valid: {results['valid_count']}, Invalid: {results['invalid_count']}")
    """
    valid_count = 0
    invalid_count = 0
    errors = []
    
    for i, example in enumerate(examples):
        try:
            validate_example_schema(example, example_id=f"example_{i}")
            valid_count += 1
        except SchemaValidationError as e:
            invalid_count += 1
            errors.append(str(e))
            
            if max_errors and len(errors) >= max_errors:
                break
    
    return {
        'valid_count': valid_count,
        'invalid_count': invalid_count,
        'total_checked': valid_count + invalid_count,
        'errors': errors,
        'all_valid': invalid_count == 0
    }


if __name__ == '__main__':
    # Simple test
    print("Testing schema validator...\n")
    
    # Valid example
    valid_ex = {
        'doc_id': 'doc_1',
        'domain_topic': 'algorithms.sorting',
        'source_text': 'Insertion sort has O(n²) worst-case',
        'generated_claim': 'Insertion sort is O(n²) in worst case',
        'gold_label': 'VERIFIED',
        'evidence_span': 'O(n²) worst-case'
    }
    
    try:
        validate_example_schema(valid_ex, 'doc_1')
        print("✓ Valid example passed")
    except SchemaValidationError as e:
        print(f"✗ Valid example failed: {e}")
    
    # Invalid example (deprecated field)
    invalid_ex = {
        'doc_id': 'doc_2',
        'claim': 'X is true',  # Wrong field name
        'gold_label': 'VERIFIED',
    }
    
    try:
        validate_example_schema(invalid_ex, 'doc_2')
        print("✗ Should have caught deprecated field")
    except SchemaValidationError as e:
        print(f"✓ Caught error as expected: {e.args[0].split(chr(10))[0]}")
    
    print("\n[OK] Schema validator working correctly")
