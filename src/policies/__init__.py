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

__all__ = [
    "is_compound_claim",
    "split_compound_claim",
    "enforce_granularity",
    "SufficiencyDecision",
    "evaluate_evidence_sufficiency",
    "THREAT_MODEL",
    "ThreatCategory"
]
