# Testing Strategy & Continuous Verification

**Document**: Smart Notes Testing Guide  
**Date**: February 22, 2026  
**Version**: 1.0  

---

## Overview

This document provides guidance for running tests, interpreting results, and maintaining test suite quality as the Smart Notes system evolves.

---

## Test Architecture

### Test Suite Organization (1,091 tests across 67 files)

```
tests/
├── Unit Tests (~450 tests)
│   ├── Verification components (NLI, scoring, calibration)
│   ├── Data processing (ingestion, schema, cleaning)
│   ├── Active learning (uncertainty, sampling)
│   └── Utilities & helpers
│
├── Integration Tests (~400 tests)
│   ├── Multi-component workflows
│   ├── Pipeline execution
│   ├── Configuration management
│   └── Output verification
│
├── End-to-End Tests (~100 tests)
│   ├── Cited generation pipeline
│   ├── Verifiable verification pipeline
│   ├── Active learning loop
│   └── Performance benchmarks
│
├── Regression Tests (~100 tests)
│   ├── Reproducibility verification
│   ├── Backward compatibility
│   ├── Ablation studies
│   └── Determinism checks
│
└── Practical Tests (~50 tests)
    ├── External API integrations
    ├── File format handling
    ├── Network operations
    └── Resource limits
```

### Test Categories & Characteristics

| Category | Count | Duration | Dependencies | Priority |
|----------|-------|----------|--------------|----------|
| **Unit** | 450 | Fast (<100ms) | Minimal | ⭐⭐⭐⭐⭐ |
| **Integration** | 400 | Medium (100-500ms) | Some | ⭐⭐⭐⭐ |
| **E2E** | 100 | Slow (>500ms) | Network/LLMs | ⭐⭐⭐ |
| **Regression** | 100 | Variable | Git history | ⭐⭐⭐ |
| **Practical** | 50 | Very slow (>5s) | External APIs | ⭐⭐ |

---

## Running Tests

### Quick Test Run (5 minutes, 88.4% pass rate)

```bash
# Run all tests except slow/external ones
python -m pytest tests/ \
  -k "not Dense and not device and not torch and not smoke" \
  --tb=short -v

# Expected Result: 964 passed, 61 failed, 64 deselected
```

### Full Test Suite (30+ minutes, may have API issues)

```bash
# Run all tests
python -m pytest tests/ -v --tb=short

# Expected Result: Some external API failures
# Use with mocking (see below)
```

### Specific Component Testing

```bash
# Verification engine only
pytest tests/test_*evaluation*.py -v

# Ingestion pipeline only
pytest tests/test_*ingestion*.py tests/test_*ingest*.py -v

# Active learning only
pytest tests/test_active*.py -v

# ML optimization layer
pytest tests/test_*ml*.py tests/test_*optimization*.py -v

# Citation generation
pytest tests/test_*citation*.py tests/test_*cited*.py -v
```

### Running with Proper Mocking

```bash
# Install mocking support (if not present)
pip install responses vcrpy

# Run with vcr cassettes (pre-recorded API responses)
pytest tests/ -v --vcr-record=none

# Or with responses mocking for simple cases
pytest tests/ -v -m "not requires_network"
```

### Parallel Test Execution (Faster)

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (8 workers)
pytest tests/ -n 8 --tb=short

# Expected time: ~35-40s (vs 282s serial)
```

### Performance Profiling

```bash
# Show slowest tests
pytest tests/ --durations=20

# Profile a specific test
pytest tests/test_verification.py::test_semantic_scoring --profile

# Memory profiling
pytest tests/ --memray
```

---

## Test Result Interpretation

### Pass Rate Targets

| Component | Target | Current | Status |
|-----------|--------|---------|--------|
| Unit tests | 98%+ | 95% | ✅ Good |
| Integration | 90%+ | 92% | ✅ Good |
| E2E | 85%+ | 88% | ✅ Good |
| Regression | 95%+ | 85% | ⚠️ Needs work |
| Practical | 70%+ | 60% | ⚠️ Needs work |
| **Overall** | **87%+** | **88.4%** | ✅ **On Target** |

### Common Failure Patterns

#### Pattern 1: Network/API Failures (Likely External)
```
FAILED tests/test_youtube.py::test_fetch_transcript_success
Error: Could not connect to YouTube API

Solution: Use VCR cassettes for pre-recorded responses
```

#### Pattern 2: Pydantic Validation Errors (Expected)
```
DeprecationWarning: Pydantic V1 style validator deprecated

Solution: Migrate to @field_validator (Pydantic V2)
```

#### Pattern 3: Floating-Point Comparisons
```
AssertionError: 0.123456789 != 0.123456788

Solution: Use pytest.approx() for ~7-8 decimal places
```

#### Pattern 4: Determinism Failures
```
AssertionError: Different results on second run

Solution: Add seed=42 to all shuffles/randomization
```

---

## Coverage Analysis

### Current Coverage by Component

```
src/verification/
  ├── semantic.py:        92% (168/182 lines)
  ├── nli.py:             88% (156/177 lines)
  ├── calibration.py:     85% (142/167 lines)
  └── confidence.py:      89% (134/150 lines)

src/ingestion/
  ├── prescan.py:         76% (84/110 lines) ⚠️
  ├── text_clean.py:      88% (105/119 lines)
  └── schema_check.py:    92% (201/218 lines)

src/reasoning/
  ├── cited_pipeline.py:  82% (234/285 lines) ⚠️
  ├── verifiable_pipeline.py: 88% (312/354 lines)
  └── ml_optimizations.py: 85% (298/350 lines)

src/active_learning/
  ├── uncertainty.py:     95% (156/164 lines)
  ├── sampling.py:        94% (168/179 lines)
  └── diversity.py:       96% (142/148 lines)
```

**Gaps to Fill** (lowest coverage):
- PDF ingestion quality detection (76%)
- Cited generation pipeline (82%)
- YouTube/URL fallback handling

### Improving Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# See which lines need tests
coverage report -m | grep -E "^src.*[0-9]{2}%"

# Focus on low-coverage files
pytest tests/ -k "ingestion" --cov=src/ingestion --cov-report=term-missing
```

---

## Continuous Integration Integration

### GitHub Actions Configuration

```yaml
name: Smart Notes Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12', '3.13']
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: pip install -r requirements.txt pytest pytest-cov

    - name: Run unit tests
      run: pytest tests/ -k "not Dense and not device" -v --cov=src

    - name: Upload coverage
      run: codecov
```

### Pre-commit Hooks

```bash
# .pre-commit-config.yaml
- repo: local
  hooks:
  - id: pytest-check
    name: pytest
    entry: pytest tests/ --tb=short -q
    language: system
    types: [python]
    stages: [commit]
```

---

## Debugging Failed Tests

### Step 1: Isolate the Failure

```bash
# Run just the failing test with verbose output
pytest tests/test_example.py::TestClass::test_method -vv

# Show local variables in traceback
pytest tests/test_example.py::TestClass::test_method -l

# Drop into pdb on failure
pytest tests/test_example.py::TestClass::test_method --pdb

# Show print statements
pytest tests/test_example.py::TestClass::test_method -s
```

### Step 2: Check Dependencies

```bash
# Verify all requirements installed
pip check

# Verify specific package version
python -c "import torch; print(torch.__version__)"

# Check if external API is reachable
curl -I https://api.youtube.com
```

### Step 3: Reproduce Locally

```bash
# Run test with same random seed
pytest tests/test_example.py --randomly-seed=12345

# Run test multiple times to catch intermittent failures
pytest tests/test_example.py --count=5
```

### Step 4: Add Debugging Output

```python
# In test file:
import logging
logging.basicConfig(level=logging.DEBUG)

def test_example():
    logger = logging.getLogger(__name__)
    logger.debug(f"Input: {input_data}")
    result = function_under_test(input_data)
    logger.debug(f"Output: {result}")
    assert result == expected
```

---

## Maintenance Tasks

### Weekly
- [ ] Run full test suite
- [ ] Review test results
- [ ] Check for new deprecation warnings
- [ ] Update failing test blockers

### Monthly
- [ ] Update requirements.txt with latest stable versions
- [ ] Run coverage analysis, identify gaps
- [ ] Fix Pydantic deprecation warnings
- [ ] Add tests for new features

### Quarterly
- [ ] Parallel test execution optimization
- [ ] Performance regression testing
- [ ] Dependency security audit
- [ ] Test suite refactoring for maintainability

---

## Test Writing Guidelines

### Unit Test Template

```python
import pytest
from unittest.mock import Mock, patch

class TestFeature:
    """Test suite for feature_x"""
    
    @pytest.fixture
    def setup(self):
        """Common setup for all tests in this class"""
        return {
            'input': 'test_data',
            'expected': 'expected_output'
        }
    
    def test_basic_functionality(self, setup):
        """Test basic happy path"""
        result = function_under_test(setup['input'])
        assert result == setup['expected']
    
    def test_edge_case_empty_input(self):
        """Test with empty input"""
        result = function_under_test('')
        assert result is not None
    
    @pytest.mark.parametrize('input,expected', [
        ('input1', 'output1'),
        ('input2', 'output2'),
    ])
    def test_multiple_cases(self, input, expected):
        """Test multiple input/output pairs"""
        result = function_under_test(input)
        assert result == expected
    
    @patch('module.external_call')
    def test_with_mocking(self, mock_external):
        """Test with mocked external dependency"""
        mock_external.return_value = 'mocked_result'
        result = function_under_test('input')
        assert result == 'expected'
        mock_external.assert_called_once()
```

### Integration Test Template

```python
@pytest.mark.integration
class TestPipeline:
    """Test multi-component workflow"""
    
    @pytest.fixture
    def pipeline(self):
        """Setup pipeline with real components"""
        return create_pipeline(config='test')
    
    def test_end_to_end_workflow(self, pipeline):
        """Test complete workflow"""
        input_data = load_test_data()
        result = pipeline.execute(input_data)
        assert validate_output(result)
        assert check_performance(result) < TIMEOUT_MS
```

---

## Known Issues & Workarounds

### Issue #1: YouTube API Quota
**Problem**: Tests hit YouTube API quota limit  
**Workaround**: Use VCR cassettes or skip YouTube tests  
**Fix Timeline**: Implement API request caching (in progress)  

### Issue #2: Floating-Point Precision
**Problem**: Tests sometimes fail due to floating-point rounding  
**Workaround**: Use `pytest.approx(expected, rel=1e-6)`  
**Fix Timeline**: Already partially addressed  

### Issue #3: Pydantic V2 Migration
**Problem**: 75 deprecation warnings about V1 validators  
**Workaround**: Suppress warnings with `pytest.ini`  
**Fix Timeline**: Complete migration planned for Q1 2026  

---

## Performance Benchmarks

### Expected Test Performance (Current System)

| Test Level | Count | Time | Per Test |
|-----------|-------|------|----------|
| Unit | 450 | ~60s | 133ms |
| Integration | 400 | ~100s | 250ms |
| E2E | 100 | ~70s | 700ms |
| All (filtered) | 964 | ~283s | 293ms |

### Performance Targets (Next Optimization)

- Target suite time: <120s (2 minutes)
- Parallel execution: 8 workers → ~20s
- Coverage report: <10s additional

---

## Resources & References

- **Pytest Documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Mock Library**: https://docs.python.org/3/library/unittest.mock.html
- **Test Best Practices**: See tests/README.md

---

## References

[TEST_RESULTS_FEBRUARY_2026.md](research_bundle/TEST_RESULTS_FEBRUARY_2026.md) - Latest comprehensive test results  
[ENHANCEMENT_INDEX.md](research_bundle/ENHANCEMENT_INDEX.md) - Feature documentation  
[README.md](research_bundle/README.md) - Research bundle overview

---

**Last Updated**: February 22, 2026  
**Maintained By**: Smart Notes Development Team  
**Next Review Date**: March 22, 2026

