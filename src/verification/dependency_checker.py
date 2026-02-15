"""
Cross-Claim Dependency Checker

This module implements lightweight dependency checking to detect claims that
reference undefined terms or concepts.

Purpose: Prevent circular or missing dependencies in knowledge graphs,
ensuring claims are properly grounded in prerequisite definitions.

Reference: Selena's research-rigor requirements for cross-claim dependencies.
"""

import re
import logging
from typing import List, Set, Dict, Tuple
from dataclasses import dataclass

from src.claims.schema import LearningClaim, ClaimType, VerificationStatus
import config

logger = logging.getLogger(__name__)


@dataclass
class DependencyWarning:
    """
    Warning about a dependency issue in a claim.
    
    Attributes:
        claim_id: ID of claim with dependency issue
        claim_text: Text of the claim
        undefined_terms: List of terms referenced but not defined
        severity: Warning severity (info, warning, error)
        suggestion: Suggested action to resolve
    """
    claim_id: str
    claim_text: str
    undefined_terms: List[str]
    severity: str = "warning"
    suggestion: str = ""


def extract_terms(text: str) -> Set[str]:
    """
    Extract key terms and entities from claim text.
    
    Extracts:
    - Capitalized terms (e.g., "Newton's Law", "Force", "Derivative")
    - Mathematical variables (single letters: x, y, k, etc.)
    - Greek letters (alpha, beta, gamma, etc.)
    - Technical terms (heuristic: words with specific patterns)
    
    Args:
        text: Claim text to analyze
    
    Returns:
        Set of extracted terms
    
    Examples:
        >>> extract_terms("Force F equals mass m times acceleration a")
        {'Force', 'F', 'm', 'a'}
        >>> extract_terms("The derivative of x^2 is 2x")
        {'derivative', 'x'}
    """
    if not text:
        return set()
    
    terms = set()
    
    # Extract capitalized words (likely proper nouns or technical terms)
    # Match words starting with capital letter
    capitalized_pattern = r'\b[A-Z][a-z]+(?:\'s)?\b'
    capitalized_terms = re.findall(capitalized_pattern, text)
    terms.update(capitalized_terms)
    
    # Extract single-letter variables (common in math/physics)
    # But avoid common articles (I, a, A)
    variable_pattern = r'\b[b-hj-zB-HJ-Z]\b'
    variables = re.findall(variable_pattern, text)
    terms.update(variables)
    
    # Extract Greek letters (written as words)
    greek_pattern = r'\b(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)\b'
    greek_letters = re.findall(greek_pattern, text, re.IGNORECASE)
    terms.update(greek_letters)
    
    # Extract technical terms (words ending in -tion, -ity, -ance, -ence, etc.)
    technical_pattern = r'\b\w+(tion|ity|ance|ence|ment|ness|ship|hood)\b'
    technical_terms = re.findall(technical_pattern, text, re.IGNORECASE)
    terms.update(technical_terms)
    
    # Extract multi-word technical phrases (capitalized words in sequence)
    phrase_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
    phrases = re.findall(phrase_pattern, text)
    terms.update(phrases)
    
    # Clean up: remove common words, articles, etc.
    stopwords = {
        'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'In', 'On', 'At',
        'Is', 'Are', 'Was', 'Were', 'Be', 'Been', 'Being', 'Have', 'Has', 'Had',
        'Do', 'Does', 'Did', 'Will', 'Would', 'Could', 'Should', 'May', 'Might',
        'Must', 'Can', 'For', 'To', 'From', 'By', 'With', 'Without', 'Of'
    }
    terms = {t for t in terms if t not in stopwords}
    
    return terms


def build_defined_terms_index(claims: List[LearningClaim]) -> Dict[str, Set[str]]:
    """
    Build index of terms defined by each claim.
    
    For DEFINITION claims, extract the term being defined (usually first capitalized term).
    For EQUATION claims, extract variables and equation name.
    
    Args:
        claims: List of claims (in order)
    
    Returns:
        Dictionary mapping claim_id to set of terms it defines
    
    Examples:
        >>> claim1 = LearningClaim(claim_type="definition", claim_text="Force is the product of mass and acceleration")
        >>> claim2 = LearningClaim(claim_type="equation", claim_text="F = ma")
        >>> index = build_defined_terms_index([claim1, claim2])
        >>> 'Force' in index[claim1.claim_id]
        True
    """
    defined_index = {}
    
    for claim in claims:
        if not claim.claim_text:
            continue
        
        defined_terms = set()
        
        if claim.claim_type == ClaimType.DEFINITION:
            # For definitions, the first capitalized term or phrase is likely the definition target
            # Pattern: "X is/are/means/refers to..."
            definition_pattern = r'^([A-Z][a-z\s]+?)\s+(is|are|means|refers to|denotes)'
            match = re.match(definition_pattern, claim.claim_text)
            if match:
                term = match.group(1).strip()
                defined_terms.add(term)
            else:
                # Fallback: just take first capitalized term
                terms = extract_terms(claim.claim_text)
                if terms:
                    # Take the first/longest capitalized term
                    sorted_terms = sorted(terms, key=len, reverse=True)
                    defined_terms.add(sorted_terms[0])
        
        elif claim.claim_type == ClaimType.EQUATION:
            # For equations, extract variable names
            # Pattern: variables on left side of = define them
            equation_match = re.search(r'(\w+)\s*=', claim.claim_text)
            if equation_match:
                var = equation_match.group(1)
                defined_terms.add(var)
            
            # Also extract equation name if present (e.g., "Newton's Second Law: F=ma")
            name_match = re.match(r"([A-Z][a-z\s']+):\s*", claim.claim_text)
            if name_match:
                name = name_match.group(1).strip()
                defined_terms.add(name)
        
        defined_index[claim.claim_id] = defined_terms
    
    return defined_index


def check_dependencies(
    claims: List[LearningClaim],
    strict_mode: bool = False
) -> List[DependencyWarning]:
    """
    Check for undefined dependencies across claims.
    
    For each claim:
    1. Extract terms referenced in the claim
    2. Check if those terms are defined in earlier claims
    3. Generate warnings for undefined terms
    
    Args:
        claims: List of claims (in order of generation)
        strict_mode: If True, treat all undefined terms as errors;
                     if False, only warn for critical undefined terms
                     (from config.STRICT_DEPENDENCY_ENFORCEMENT)
    
    Returns:
        List of dependency warnings
    
    Side effects:
        - Logs INFO for dependency analysis
        - Logs WARNING for each undefined term detected
    """
    if strict_mode is None:
        strict_mode = config.STRICT_DEPENDENCY_ENFORCEMENT
    
    logger.info(f"Checking dependencies for {len(claims)} claims (strict_mode={strict_mode})")
    
    warnings = []
    defined_terms = set()
    defined_index = build_defined_terms_index(claims)
    
    for i, claim in enumerate(claims):
        if not claim.claim_text:
            continue
        
        # Add terms defined by this claim to global set
        if claim.claim_id in defined_index:
            defined_terms.update(defined_index[claim.claim_id])
        
        # Extract terms referenced in this claim
        referenced_terms = extract_terms(claim.claim_text)
        
        # Check which terms are undefined
        undefined = referenced_terms - defined_terms
        
        # Filter out very common terms that don't need definition
        # E.g., "equation", "value", "result", "example"
        common_terms = {
            'equation', 'value', 'result', 'example', 'definition', 'term',
            'concept', 'property', 'method', 'process', 'system', 'function',
            'relation', 'relationship', 'statement', 'proof', 'theorem',
            'formula', 'expression', 'variable', 'constant', 'parameter'
        }
        undefined = {t for t in undefined if t.lower() not in common_terms}
        
        if undefined:
            severity = "error" if strict_mode else "warning"
            suggestion = (
                f"Define these terms before claim {i+1}, or verify they are standard notation."
                if strict_mode else
                f"Consider defining these terms explicitly for clarity."
            )
            
            warning = DependencyWarning(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text[:100] + "..." if len(claim.claim_text) > 100 else claim.claim_text,
                undefined_terms=sorted(undefined),
                severity=severity,
                suggestion=suggestion
            )
            warnings.append(warning)
            
            logger.warning(
                f"Claim {i+1} ({claim.claim_id[:8]}...): "
                f"References {len(undefined)} undefined terms: {', '.join(sorted(undefined)[:5])}"
            )
    
    logger.info(f"Dependency check complete: {len(warnings)} warnings generated")
    
    return warnings


def apply_dependency_enforcement(
    claims: List[LearningClaim],
    warnings: List[DependencyWarning],
    downgrade_to_low_confidence: bool = False
) -> List[LearningClaim]:
    """
    Apply dependency enforcement by downgrading claims with undefined dependencies.
    
    If downgrade_to_low_confidence is True (or config.STRICT_DEPENDENCY_ENFORCEMENT):
    - Claims with undefined dependencies are downgraded to LOW_CONFIDENCE
    - Rejection reason set to DEPENDENCY_REQUIRED
    
    Args:
        claims: List of claims
        warnings: List of dependency warnings from check_dependencies()
        downgrade_to_low_confidence: Whether to downgrade claims with warnings
    
    Returns:
        Updated list of claims (some may be downgraded)
    
    Side effects:
        - Updates claim.status for claims with warnings
        - Updates claim.confidence (reduced by 0.3)
        - Adds dependency_warning to claim.metadata
    """
    if downgrade_to_low_confidence is None:
        downgrade_to_low_confidence = config.STRICT_DEPENDENCY_ENFORCEMENT
    
    if not downgrade_to_low_confidence:
        logger.info("Dependency enforcement disabled; warnings logged only")
        return claims
    
    logger.info(f"Applying dependency enforcement to {len(warnings)} claims with warnings")
    
    # Build map of claim_id -> warning
    warning_map = {w.claim_id: w for w in warnings}
    
    downgraded_count = 0
    for claim in claims:
        if claim.claim_id in warning_map:
            warning = warning_map[claim.claim_id]
            
            # Only downgrade VERIFIED claims (don't affect already rejected/low-confidence)
            if claim.status == VerificationStatus.VERIFIED:
                claim.status = VerificationStatus.LOW_CONFIDENCE
                claim.confidence = max(0.0, claim.confidence - 0.3)  # Penalize confidence
                claim.metadata["dependency_warning"] = {
                    "undefined_terms": warning.undefined_terms,
                    "severity": warning.severity,
                    "suggestion": warning.suggestion
                }
                downgraded_count += 1
                logger.info(f"Downgraded claim {claim.claim_id[:8]} due to undefined dependencies")
    
    logger.info(f"Dependency enforcement complete: {downgraded_count} claims downgraded")
    
    return claims
