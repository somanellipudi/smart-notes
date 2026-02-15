"""
Verification module for cross-claim consistency checks.

This module provides tools for checking dependencies and consistency
across multiple claims.
"""

from .dependency_checker import (
    DependencyWarning,
    extract_terms,
    build_defined_terms_index,
    check_dependencies,
    apply_dependency_enforcement
)

__all__ = [
    "DependencyWarning",
    "extract_terms",
    "build_defined_terms_index",
    "check_dependencies",
    "apply_dependency_enforcement"
]
