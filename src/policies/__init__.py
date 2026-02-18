"""
Policy modules for research-rigor upgrades.

This package contains deterministic policy implementations for:
- Claim granularity enforcement (atomic claims)
- Evidence sufficiency decision rules
- Threat model documentation
"""

from .granularity_policy import (
    is_compound_claim,
    split_compound_claim,
    enforce_granularity
)
from .evidence_policy import (
    SufficiencyDecision,
    evaluate_evidence_sufficiency
)
from .threat_model import THREAT_MODEL, ThreatCategory
from .domain_profiles import DomainProfile, DOMAIN_PROFILES, get_domain_profile
from .verification_policy import VerificationDecision, evaluate_claim, tag_claim_type

__all__ = [
    "is_compound_claim",
    "split_compound_claim",
    "enforce_granularity",
    "SufficiencyDecision",
    "evaluate_evidence_sufficiency",
    "THREAT_MODEL",
    "ThreatCategory",
    "DomainProfile",
    "DOMAIN_PROFILES",
    "get_domain_profile",
    "VerificationDecision",
    "evaluate_claim",
    "tag_claim_type"
]
