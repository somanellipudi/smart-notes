"""
Pytest configuration and fixtures for Smart Notes evaluation tests.
"""

from pathlib import Path


def pytest_ignore_collect(path):
    """Ignore problematic non-Python test-like files in repo root and artifacts."""
    p = Path(str(path))
    name = p.name
    # Ignore top-level test text files and artifacts binary dumps
    if name in {"test_results_latest.txt", "test_output.txt"}:
        return True
    # Ignore any files under artifacts that are not python tests
    if "artifacts" in p.parts:
        return True
    return False


import pytest
import numpy as np
import tempfile
from src.evaluation.synthetic_data import (
    generate_synthetic_csclaimbench,
    generate_synthetic_calibration_data,
    generate_synthetic_fever_like,
    generate_synthetic_extended_csclaimbench,
)
from src.config.verification_config import VerificationConfig


def pytest_configure(config):
    """Register custom pytest marks."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark test"
    )


# ==================== Test Fixtures (Deterministic) ====================


@pytest.fixture
def verification_config() -> VerificationConfig:
    """
    Deterministic verification config (seed=42).
    
    Fixture provides default config for all tests.
    """
    cfg = VerificationConfig(
        random_seed=42,
        deployment_mode="verifiable",
    )
    return cfg


@pytest.fixture
def verification_config_full_optimization() -> VerificationConfig:
    """Full optimization mode config."""
    return VerificationConfig(
        random_seed=42,
        deployment_mode="full_default",
    )


@pytest.fixture
def verification_config_minimal() -> VerificationConfig:
    """Minimal deployment config (75% cost reduction)."""
    return VerificationConfig(
        random_seed=42,
        deployment_mode="minimal_deployment",
    )


@pytest.fixture
def synthetic_csclaimbench_records():
    """
    ⚠️ SYNTHETIC PLACEHOLDER (300 records, seed=42)
    
    For unit testing only. Do NOT use for scientific claims.
    """
    return generate_synthetic_csclaimbench(n_samples=300, seed=42)


@pytest.fixture
def synthetic_csclaimbench_extended():
    """
    ⚠️ SYNTHETIC PLACEHOLDER (560 records, seed=42)
    
    Extended set for scalability testing. Do NOT use for scientific claims.
    """
    return generate_synthetic_extended_csclaimbench(n_samples=560, seed=42)


@pytest.fixture
def synthetic_fever_records():
    """
    ⚠️ SYNTHETIC PLACEHOLDER (200 records, seed=42)
    
    FEVER-schema-like data for transfer evaluation testing.
    """
    return generate_synthetic_fever_like(n_samples=200, seed=42)


@pytest.fixture
def synthetic_calibration_data():
    """
    ⚠️ SYNTHETIC PLACEHOLDER (100 samples, seed=42)
    
    Deterministic calibration data: (confidences, labels).
    
    Returns:
        Tuple of (confidences: np.ndarray, labels: np.ndarray)
    """
    confidences, labels = generate_synthetic_calibration_data(n_samples=100, seed=42)
    return confidences, labels


@pytest.fixture
def temp_output_dir():
    """
    Temporary output directory for test-generated artifacts.
    
    Automatically cleaned up after test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def synthetic_csclaimbench_jsonl(temp_output_dir):
    """
    Create a temporary synthetic CSClaimBench JSONL file.
    
    ⚠️ SYNTHETIC PLACEHOLDER
    
    Returns path to generated JSONL file.
    """
    outpath = temp_output_dir / "csclaimbench_synthetic.jsonl"
    generate_synthetic_csclaimbench(n_samples=300, seed=42, outpath=str(outpath))
    return str(outpath)


@pytest.fixture
def synthetic_fever_jsonl(temp_output_dir):
    """
    Create a temporary synthetic FEVER JSONL file.
    
    ⚠️ SYNTHETIC PLACEHOLDER
    
    Returns path to generated JSONL file.
    """
    outpath = temp_output_dir / "fever_synthetic.jsonl"
    generate_synthetic_fever_like(n_samples=200, seed=42, outpath=str(outpath))
    return str(outpath)
