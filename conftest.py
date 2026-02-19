"""
Pytest configuration and fixtures for Smart Notes evaluation tests.
"""

import pytest


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
