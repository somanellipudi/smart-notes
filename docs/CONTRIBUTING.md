# Contributing to Smart Notes

Welcome to Smart Notes! This guide explains how to contribute tests, datasets, documentation, and code.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Code Organization](#code-organization)
3. [Adding Tests](#adding-tests)
4. [Adding Benchmark Datasets](#adding-benchmark-datasets)
5. [Documentation Standards](#documentation-standards)
6. [Running Tests Locally](#running-tests-locally)
7. [Submitting Changes](#submitting-changes)

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- Virtual environment

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/smart-notes.git
cd smart-notes

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov  # For testing

# Verify setup
pytest tests/ -v --co  # Show test collection
```

---

## Code Organization

### Key Directories

```
src/                  → Application code (organized by module)
tests/               → Test code (organized by category)
evaluation/          → Research benchmarks and results
docs/                → Documentation
examples/            → Usage examples
scripts/             → Executable scripts
```

### Adding New Code

**Application Code**:
- Add to appropriate `src/` module  
- Follow existing patterns and naming
- Include docstrings

**Utilities/Scripts**:
- Add to `scripts/` for analysis and benchmarking
- Add to `examples/` for usage demonstrations

---

## Adding Tests

### Test File Organization

Tests are organized by category in `tests/` directory:

```
tests/
├── test_evaluation_comprehensive.py    # Benchmark component tests (50+)
├── test_integration_evaluation.py      # End-to-end integration (40+)
├── test_benchmark_format_validation.py # Dataset schema validation
├── test_ablation_runner_smoke.py       # Ablation runner tests
├── test_artifact_roundtrip.py          # Core functionality
├── test_pdf_*.py                       # PDF processing
├── test_ingestion_*.py                 # Data ingestion
├── test_retrieval_*.py                 # Evidence retrieval
└── archived/                           # Old/deprecated tests
    └── README.md                       # Archival rationale
```

### Test File Naming Convention

```python
test_<category>_<component>.py     # General format
test_evaluation_comprehensive.py   # Comprehensive evaluation tests
test_integration_pdf_url.py        # Integration: PDF from URL
```

### Creating a New Test File

1. **Create file** in `tests/` with name `test_<component>.py`

2. **Use pytest structure**:

```python
"""
Test Module: <Component Description>

Tests for <what you're testing>. Organized into logical test classes.
"""

import pytest
from pathlib import Path

class TestComponentName:
    """Test class for logical grouping."""
    
    @pytest.fixture
    def setup_resource(self):
        """Setup fixture for tests."""
        resource = create_resource()
        yield resource
        cleanup_resource()
    
    def test_basic_functionality(self, setup_resource):
        """Test basic behavior."""
        result = setup_resource.method()
        assert result == expected_value
    
    @pytest.mark.slow
    def test_integration_workflow(self):
        """Integration test (skip in fast runs)."""
        # Test that takes > 1 second
        pass
```

3. **Add to test discovery** (pytest finds it automatically)

4. **Run locally**:
```bash
pytest tests/test_<component>.py -v
```

### Test Best Practices

- ✅ Use fixtures for setup/teardown
- ✅ Use descriptive test names (`test_<action>_<expected_result>`)
- ✅ Group related tests in classes
- ✅ Mark slow tests with `@pytest.mark.slow`
- ✅ Include docstrings explaining what is tested
- ✅ Use assertions that are easy to debug
- ❌ Don't hardcode paths (use fixtures)
- ❌ Don't leave resource leaks (cleanup in fixtures)

---

## Adding Benchmark Datasets

### Dataset Purpose

Benchmark datasets in `evaluation/cs_benchmark/` are used for:
- Ablation studies (comparing configurations)
- Robustness evaluation (diverse inputs)
- Publication results (reproducible benchmarks)

### Creating a New Dataset

1. **Choose a purpose** from existing datasets:
   - **Core**: General-purpose, balanced
   - **Hard**: Challenging cases  (subtle errors, multi-hop reasoning)
   - **Easy**: Simple cases (clear evidence, unambiguous labels)
   - **Domain-specific**: Per-category analysis
   - **Adversarial**: Paraphrasing, noise injection

2. **Create JSONL file** in `evaluation/cs_benchmark/`:

```bash
# Naming convention
cs_benchmark_<purpose>.jsonl

# Example
cs_benchmark_multilingual.jsonl      # New multilingual dataset
```

3. **Follow schema** (all fields required):

```json
{
  "doc_id": "unique_id_001",
  "domain_topic": "category.subcategory",
  "source_text": "original document text",
  "generated_claim": "claim to verify",
  "gold_label": "VERIFIED|REJECTED|LOW_CONFIDENCE",
  "evidence_span": "exact text span from source_text (or empty)",
  "difficulty": "easy|medium|hard",
  "source_type": "textbook|paper|wikipedia|doc|tutorial",
  "claim_type": "factual|definition|performance|relationship",
  "reasoning_type": "direct|implicit|multi_hop|arithmetic",
  "metadata": {
    "creation_date": "2026-02-17",
    "verifier": "annotator_name",
    "notes": "optional comments"
  }
}
```

4. **Validate dataset**:

```bash
# Run validation tests
pytest tests/test_benchmark_format_validation.py -v

# Or run specific checks
python -c "
import json
with open('evaluation/cs_benchmark/your_dataset.jsonl') as f:
    examples = [json.loads(line) for line in f]
    print(f'Loaded {len(examples)} examples')
    # Verify fields, labels, etc.
"
```

5. **Update documentation**:

Edit `evaluation/cs_benchmark/README_DATASETS.md`:
- Add your dataset to the table
- Describe purpose and characteristics
- List size and key properties

6. **Run benchmark** on your dataset:

```bash
python scripts/run_cs_benchmark.py \
    --dataset evaluation/cs_benchmark/your_dataset.jsonl \
    --output-dir evaluation/results/your_analysis
```

### Dataset Validation Checklist

- [ ] All JSONL lines are valid JSON
- [ ] All required fields present
- [ ] gold_label in {VERIFIED, REJECTED, LOW_CONFIDENCE}
- [ ] difficulty in {easy, medium, hard}
- [ ] If VERIFIED: evidence_span should be non-empty
- [ ] If REJECTED: evidence_span can be empty
- [ ] Unique doc_ids across all examples
- [ ] No extremely long texts (>10k chars)
- [ ] No sensitive/copyrighted content

---

## Documentation Standards

### File Documentation

**Docstrings for modules**:
```python
"""
Module: claims.nli_verifier

Provides NLI-based claim verification using RoBERTa-large-MNLI.

Classes:
    NLIVerifier: Main verification engine
    
Key functions:
    verify(claim, evidence): Single pair verification
    verify_batch(pairs): Batch verification
    
Example:
    >>> verifier = NLIVerifier()
    >>> result = verifier.verify("A is B", "B is A")
"""
```

**Docstrings for functions**:
```python
def verify_batch(
    self,
    pairs: List[Tuple[str, str]]
) -> List[NLIResult]:
    """
    Verify multiple claim-evidence pairs in batch.
    
    Optimized for throughput with configurable batch size.
    
    Args:
        pairs: List of (claim, evidence) tuples
        
    Returns:
        List of NLIResult objects in same order as input
        
    Example:
        >>> pairs = [("A", "B"), ("C", "D")]
        >>> results = verifier.verify_batch(pairs)
    """
```

### Markdown Documentation

**Structure**:
```markdown
# Title

## Overview
- What this does
- Why it matters

## Usage

```bash
command to run
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|

## Examples

### Example 1: Basic usage
## Troubleshooting

### Problem: ...
Solution: ...
```

---

## Running Tests Locally

### All Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Stop on first failure
pytest tests/ -x

# Run only failed tests
pytest tests/ --lf
```

### By Category

```bash
# Evaluation tests
pytest tests/test_evaluation_*.py -v

# PDF processing
pytest tests/test_pdf_*.py -v

# Specific test
pytest tests/test_benchmark_format_validation.py::TestBenchmarkFormatValidation::test_dataset_exists -v
```

### Fast vs Slow

```bash
# Skip slow tests (< 10 seconds)
pytest tests/ -v -m "not slow"

# Only slow tests
pytest tests/ -v -m "slow"
```

### With Markers

```bash
# Define in test
@pytest.mark.slow
@pytest.mark.integration
def test_long_running():
    pass

# Run with markers
pytest tests/ -m "not slow"
pytest tests/ -m "integration"
```

---

## Submitting Changes

### Before Submitting

1. **Run tests locally**:
   ```bash
   pytest tests/ -v --tb=short
   ```

2. **Check code style**:
   ```bash
   # Optional: use black for formatting
   black src/ tests/
   ```

3. **Update documentation**:
   - Add docstrings to new functions
   - Update relevant README files
   - Update PROJECT_STRUCTURE.md if adding files

4. **Verify no breaking changes**:
   ```bash
   pytest tests/ -v  # All tests should pass
   ```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`

**Examples**:
```
test: add 20 hard benchmark examples

Adds challenging verification cases with subtle errors
and multi-hop reasoning. Improves robustness evaluation.

Closes #123

---

feat: add multilingual benchmark dataset

Adds 25-example dataset across English, Spanish, French.
Updates evaluation infrastructure for language diversity.

docs: clarify test organization in PROJECT_STRUCTURE.md

Restructures section on test file locations and naming
conventions for clarity.
```

### Pull Request Template

```markdown
## Description
Brief summary of changes

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Documentation update
- [ ] Test addition
- [ ] Refactoring

## Tests
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] New tests added for new functionality
- [ ] Coverage maintained (>80%)

## Documentation
- [ ] Updated relevant documentation
- [ ] Updated PROJECT_STRUCTURE.md if applicable
- [ ] Added docstrings

## Checklist
- [ ] Code follows project style
- [ ] No hard-coded paths
- [ ] No breaking changes
```

---

## Getting Help

- **Discussion**: Check existing GitHub issues
- **Documentation**: Read `docs/PROJECT_STRUCTURE.md`
- **Examples**: See `examples/` directory
- **Tests**: Look at similar tests in `tests/`

---

## Code of Conduct

- Be respectful and inclusive
- Assume good intent
- Provide constructive feedback
- Help others succeed

---

## License

By contributing to Smart Notes, you agree that your contributions will be licensed under the same license as the project.

---

*Last Updated: 2026-02-17*
