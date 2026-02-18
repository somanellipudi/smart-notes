"""
Domain profiles for domain-scoped verification rules.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DomainProfile:
    """
    Domain-specific validation profile for research-grade verifiability.

    Attributes:
        name: Domain identifier
        display_name: Human-readable name
        description: Domain description
        allowed_claim_types: Claim types relevant to this domain
        evidence_type_expectations: Expected evidence types per claim type
        require_units: Whether unit checking is required
        require_proof_steps: Whether proof-step strictness is enforced
        require_pseudocode: Whether pseudocode checks are required
        require_equations: Whether equations must be present
        strict_dependencies: Whether to enforce strict dependency checking
    """
    name: str
    display_name: str
    description: str
    allowed_claim_types: List[str]
    evidence_type_expectations: Dict[str, List[str]]
    require_units: bool = False
    require_proof_steps: bool = False
    require_pseudocode: bool = False
    require_equations: bool = False
    strict_dependencies: bool = False


DOMAIN_PROFILES: Dict[str, DomainProfile] = {
    "physics": DomainProfile(
        name="physics",
        display_name="Physics",
        description="Physics domain with equations, units, and physical laws",
        allowed_claim_types=["definition", "equation", "example", "misconception"],
        evidence_type_expectations={
            "definition": ["transcript", "notes", "external"],
            "equation": ["transcript", "notes", "external", "equation"],
            "example": ["transcript", "notes", "external"],
            "misconception": ["transcript", "notes"]
        },
        require_units=True,
        require_proof_steps=False,
        require_pseudocode=False,
        require_equations=True,
        strict_dependencies=False
    ),
    "discrete_math": DomainProfile(
        name="discrete_math",
        display_name="Discrete Mathematics",
        description="Discrete math domain with definitions, proofs, and formal logic",
        allowed_claim_types=["definition", "example", "misconception"],
        evidence_type_expectations={
            "definition": ["transcript", "notes", "external"],
            "example": ["transcript", "notes", "external"],
            "misconception": ["transcript", "notes"]
        },
        require_units=False,
        require_proof_steps=True,
        require_pseudocode=False,
        require_equations=False,
        strict_dependencies=True
    ),
    "algorithms": DomainProfile(
        name="algorithms",
        display_name="Algorithms & Data Structures",
        description="Algorithms domain with pseudocode and complexity analysis",
        allowed_claim_types=["definition", "algorithm_step", "complexity", "invariant"],
        evidence_type_expectations={
            "definition": ["transcript", "notes", "external"],
            "algorithm_step": ["transcript", "notes", "external"],
            "complexity": ["transcript", "notes", "external"],
            "invariant": ["transcript", "notes", "external"]
        },
        require_units=False,
        require_proof_steps=False,
        require_pseudocode=True,
        require_equations=False,
        strict_dependencies=False
    ),
    "cs": DomainProfile(
        name="cs",
        display_name="CS Algorithms & Data Structures",
        description="CS algorithms domain with pseudocode, invariants, and complexity",
        allowed_claim_types=["definition", "algorithm_step", "complexity", "invariant"],
        evidence_type_expectations={
            "definition": ["transcript", "notes", "external"],
            "algorithm_step": ["transcript", "notes", "external"],
            "complexity": ["transcript", "notes", "external"],
            "invariant": ["transcript", "notes", "external"]
        },
        require_units=False,
        require_proof_steps=False,
        require_pseudocode=True,
        require_equations=False,
        strict_dependencies=False
    )
}


def get_domain_profile(domain_name: str, default_domain: str = "physics") -> DomainProfile:
    if domain_name is None:
        domain_name = default_domain

    if domain_name not in DOMAIN_PROFILES:
        raise ValueError(
            f"Invalid domain: {domain_name}. "
            f"Valid domains: {', '.join(DOMAIN_PROFILES.keys())}"
        )

    return DOMAIN_PROFILES[domain_name]
