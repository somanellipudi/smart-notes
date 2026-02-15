"""
Threat Model for Verifiable Mode

This module defines the threat model for research-grade verifiable mode,
documenting in-scope and out-of-scope threats.

Purpose: Explicit threat model enables reproducible research and sets
clear boundaries for what the system can and cannot verify.

Reference: Selena's research-rigor requirements for threat model documentation.
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class ThreatCategory(str, Enum):
    """Categories of threats to verifiability."""
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class Threat:
    """
    A specific threat to verifiability.
    
    Attributes:
        name: Short threat identifier
        description: Detailed description of the threat
        category: Whether threat is in-scope or out-of-scope
        mitigation: How the system addresses this threat (for in-scope)
        rationale: Why this threat is out-of-scope (for out-of-scope)
    """
    name: str
    description: str
    category: ThreatCategory
    mitigation: str = ""
    rationale: str = ""


# Define threat model
THREAT_MODEL: Dict[str, Threat] = {
    # ==================== IN-SCOPE THREATS ====================
    "unsupported_claims": Threat(
        name="Unsupported Claims (Hallucinations)",
        description=(
            "AI-generated claims that lack sufficient evidence in source material. "
            "This is the primary target of verifiable mode."
        ),
        category=ThreatCategory.IN_SCOPE,
        mitigation=(
            "Evidence-first generation: require evidence retrieval before claim text generation. "
            "Sufficiency policy: enforce minimum entailment probability and independent sources. "
            "Automatic rejection of claims with NO_EVIDENCE or INSUFFICIENT_CONFIDENCE."
        )
    ),
    
    "scope_creep_overgeneralization": Threat(
        name="Scope Creep and Overgeneralization",
        description=(
            "Claims that extend beyond the evidence scope or overgeneralize from limited examples. "
            "E.g., asserting 'all X have property Y' when evidence only covers specific cases."
        ),
        category=ThreatCategory.IN_SCOPE,
        mitigation=(
            "Claim granularity policy: enforce atomic claims to prevent bundling assertions. "
            "Evidence sufficiency: require multiple independent sources to support generalizations. "
            "NLI verification: detect when claim semantically extends beyond evidence."
        )
    ),
    
    "misinterpreted_equations_units": Threat(
        name="Misinterpreted Equations and Units",
        description=(
            "Incorrect interpretation of mathematical equations, unit conversions, or physical constants. "
            "Domain-specific to physics/engineering contexts."
        ),
        category=ThreatCategory.IN_SCOPE,
        mitigation=(
            "Domain profiles: physics profile enables unit checking requirements. "
            "Evidence validation: require equation claims to match exact equation syntax in sources. "
            "Cross-claim dependency checking: detect inconsistent unit usage across claims."
        )
    ),
    
    "contradiction_across_sources": Threat(
        name="Contradiction Across Sources",
        description=(
            "Conflicting information from multiple sources leading to inconsistent claims. "
            "E.g., lecture transcript says X, but textbook says Y."
        ),
        category=ThreatCategory.IN_SCOPE,
        mitigation=(
            "Evidence sufficiency policy: flag HIGH_CONTRADICTION as LOW_CONFIDENCE. "
            "Conflict detection: NLI checks for entailment vs. contradiction across evidence. "
            "Multi-source requirement: require agreement across independent sources for VERIFIED status."
        )
    ),
    
    "circular_dependencies": Threat(
        name="Circular or Missing Dependencies",
        description=(
            "Claims that reference undefined terms or create circular definition chains. "
            "E.g., defining A using B, but B is never defined or B is defined using A."
        ),
        category=ThreatCategory.IN_SCOPE,
        mitigation=(
            "Cross-claim dependency checker: detect undefined term references. "
            "Domain profiles: discrete_math profile enforces strict dependency checking. "
            "Warnings list: surface dependency issues in UI and export metadata."
        )
    ),
    
    # ==================== OUT-OF-SCOPE THREATS ====================
    "ocr_noise": Threat(
        name="OCR Noise and Errors",
        description=(
            "Errors introduced during Optical Character Recognition (OCR) of scanned documents. "
            "E.g., '0' misread as 'O', mathematical symbols corrupted."
        ),
        category=ThreatCategory.OUT_OF_SCOPE,
        rationale=(
            "OCR is a preprocessing step outside verifiable mode boundaries. "
            "System assumes OCR quality is handled externally (e.g., using high-quality OCR tools, "
            "manual review, or confidence-filtered OCR outputs). "
            "If OCR quality is poor, evidence retrieval will fail to match claims, "
            "resulting in NO_EVIDENCE rejection (acceptable fallback)."
        )
    ),
    
    "transcription_errors": Threat(
        name="Audio Transcription Errors",
        description=(
            "Errors introduced during speech-to-text transcription (e.g., Whisper). "
            "E.g., mishearing 'mass' as 'mask', missing technical terms."
        ),
        category=ThreatCategory.OUT_OF_SCOPE,
        rationale=(
            "Transcription quality is a preprocessing concern. "
            "System assumes transcription is of acceptable quality (e.g., using domain-adapted "
            "speech recognition, manual correction, or confidence thresholding). "
            "Poor transcription will lead to evidence retrieval failures, triggering rejection."
        )
    ),
    
    "domain_ambiguity": Threat(
        name="Domain Ambiguity",
        description=(
            "Ambiguous content that could belong to multiple domains, leading to incorrect "
            "domain profile selection. E.g., 'graph' could mean graph theory (discrete math) "
            "or function plot (calculus)."
        ),
        category=ThreatCategory.OUT_OF_SCOPE,
        rationale=(
            "Domain selection is a user responsibility. "
            "Users must select the appropriate domain profile based on course/lecture context. "
            "System does not attempt automatic domain classification. "
            "Future work: add domain detection heuristics or multi-domain support."
        )
    ),
    
    "adversarial_inputs": Threat(
        name="Adversarial Inputs",
        description=(
            "Intentionally crafted malicious inputs designed to exploit system vulnerabilities. "
            "E.g., prompt injection, evidence poisoning, jailbreaks."
        ),
        category=ThreatCategory.OUT_OF_SCOPE,
        rationale=(
            "Verifiable mode is designed for educational use with honest users. "
            "Adversarial robustness is not a primary design goal. "
            "If adversarial inputs are a concern, additional layers (input validation, "
            "content moderation, rate limiting) should be added externally."
        )
    ),
    
    "external_context_quality": Threat(
        name="External Context Quality",
        description=(
            "Low-quality or incorrect external context (e.g., outdated textbooks, unreliable websites). "
            "System treats external context as trusted sources."
        ),
        category=ThreatCategory.OUT_OF_SCOPE,
        rationale=(
            "Users are responsible for providing high-quality external context. "
            "System does not validate external source credibility. "
            "Future work: add source reliability priors or citation quality scoring."
        )
    )
}


def get_in_scope_threats() -> List[Threat]:
    """
    Get all in-scope threats.
    
    Returns:
        List of threats that the system actively mitigates
    """
    return [
        threat for threat in THREAT_MODEL.values()
        if threat.category == ThreatCategory.IN_SCOPE
    ]


def get_out_of_scope_threats() -> List[Threat]:
    """
    Get all out-of-scope threats.
    
    Returns:
        List of threats that the system does not address
    """
    return [
        threat for threat in THREAT_MODEL.values()
        if threat.category == ThreatCategory.OUT_OF_SCOPE
    ]


def format_threat_model(markdown: bool = True) -> str:
    """
    Format threat model as human-readable text.
    
    Args:
        markdown: If True, format as Markdown; otherwise plain text
    
    Returns:
        Formatted threat model documentation
    """
    if markdown:
        output = "# Threat Model for Verifiable Mode\n\n"
        
        output += "## In-Scope Threats\n\n"
        output += "These threats are actively mitigated by the system:\n\n"
        for threat in get_in_scope_threats():
            output += f"### {threat.name}\n\n"
            output += f"**Description:** {threat.description}\n\n"
            output += f"**Mitigation:** {threat.mitigation}\n\n"
        
        output += "## Out-of-Scope Threats\n\n"
        output += "These threats are not addressed by the system:\n\n"
        for threat in get_out_of_scope_threats():
            output += f"### {threat.name}\n\n"
            output += f"**Description:** {threat.description}\n\n"
            output += f"**Rationale:** {threat.rationale}\n\n"
    else:
        output = "THREAT MODEL FOR VERIFIABLE MODE\n\n"
        
        output += "IN-SCOPE THREATS\n"
        output += "================\n\n"
        for threat in get_in_scope_threats():
            output += f"{threat.name}\n"
            output += f"Description: {threat.description}\n"
            output += f"Mitigation: {threat.mitigation}\n\n"
        
        output += "OUT-OF-SCOPE THREATS\n"
        output += "====================\n\n"
        for threat in get_out_of_scope_threats():
            output += f"{threat.name}\n"
            output += f"Description: {threat.description}\n"
            output += f"Rationale: {threat.rationale}\n\n"
    
    return output


def get_threat_model_summary() -> Dict[str, any]:
    """
    Get structured summary of threat model for metadata export.
    
    Returns:
        Dictionary with threat model summary
    """
    return {
        "in_scope_count": len(get_in_scope_threats()),
        "out_of_scope_count": len(get_out_of_scope_threats()),
        "in_scope_threats": [t.name for t in get_in_scope_threats()],
        "out_of_scope_threats": [t.name for t in get_out_of_scope_threats()],
        "version": "1.0",
        "last_updated": "2026-02-13"
    }
