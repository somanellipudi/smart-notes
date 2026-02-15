"""
Claims module for Verifiable Mode.

This module implements claim-based, evidence-grounded generation
for research-oriented educational content extraction.
"""

from src.claims.schema import (
    LearningClaim,
    ClaimType,
    VerificationStatus,
    RejectionReason,
    EvidenceItem,
    ClaimCollection
)
from src.claims.extractor import ClaimExtractor

__all__ = [
    "LearningClaim",
    "ClaimType",
    "VerificationStatus",
    "RejectionReason",
    "EvidenceItem",
    "ClaimCollection",
    "ClaimExtractor"
]
