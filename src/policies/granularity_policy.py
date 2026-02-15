"""
Claim Granularity Policy - Enforce Atomic Claims

This module implements deterministic rules to split compound claims into
atomic, individually-verifiable propositions.

Policy: Each claim must be atomic (max 1 proposition).

Reference: Selena's research-rigor requirements for formal claim granularity.
"""

import re
import logging
from typing import List
from src.claims.schema import LearningClaim, ClaimType

logger = logging.getLogger(__name__)


def is_compound_claim(text: str) -> bool:
    """
    Detect if a claim contains multiple propositions (compound claim).
    
    Uses heuristics based on:
    - Multiple sentences (more than 1)
    - Conjunctions that connect independent clauses ("and", "or", "but")
    - Semicolons separating independent statements
    - Multiple equations
    - Multiple "because"/"therefore" indicators
    
    Args:
        text: Claim text to analyze
    
    Returns:
        True if claim appears to be compound, False if atomic
    
    Examples:
        >>> is_compound_claim("Force equals mass times acceleration.")
        False
        >>> is_compound_claim("Force equals mass times acceleration and velocity is distance over time.")
        True
        >>> is_compound_claim("The limit exists. It equals zero.")
        True
    """
    if not text or len(text.strip()) < 10:
        return False
    
    text = text.strip()
    
    # Count sentences (using period, exclamation, or question mark)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > 1:
        logger.debug(f"Compound claim detected: {len(sentences)} sentences")
        return True
    
    # Count semicolons (separate independent clauses)
    if ';' in text:
        logger.debug("Compound claim detected: semicolon present")
        return True
    
    # Count coordinating conjunctions connecting independent clauses
    # Simple heuristic: lowercase "and", "or", "but" surrounded by spaces
    # (to avoid matching "standard", "factor", "button")
    conjunction_pattern = r'\s+(and|or|but)\s+'
    conjunctions = re.findall(conjunction_pattern, text, re.IGNORECASE)
    if len(conjunctions) >= 1:
        # Additional check: ensure it's not just a list (e.g., "A and B" in simple noun phrase)
        # If the text has commas near the conjunction, it might be a list
        # For now, accept any conjunction as potential compound
        logger.debug(f"Compound claim detected: {len(conjunctions)} conjunctions")
        return True
    
    # Count multiple equations (pattern: "= ... =" or multiple "f(x) =" patterns)
    equation_pattern = r'[a-zA-Z_]\w*\s*=|=\s*\d'
    equations = re.findall(equation_pattern, text)
    if len(equations) > 1:
        logger.debug(f"Compound claim detected: {len(equations)} equations")
        return True
    
    # Count causal/logical connectors (because, therefore, thus, hence)
    causal_pattern = r'\b(because|therefore|thus|hence|so)\b'
    causals = re.findall(causal_pattern, text, re.IGNORECASE)
    if len(causals) >= 2:
        logger.debug(f"Compound claim detected: {len(causals)} causal connectors")
        return True
    
    return False


def split_compound_claim(text: str, max_splits: int = 5) -> List[str]:
    """
    Split a compound claim into atomic sub-claims.
    
    Splitting strategy:
    1. Split by multiple sentences (period boundaries)
    2. Split by semicolons
    3. Split by coordinating conjunctions ("and", "or", "but") when they connect independent clauses
    
    Note: This is a heuristic implementation. More sophisticated NLP parsing
    (dependency parsing, constituency parsing) could improve accuracy.
    
    Args:
        text: Compound claim text
        max_splits: Maximum number of atomic claims to return (safety limit)
    
    Returns:
        List of atomic claim texts
    
    Examples:
        >>> split_compound_claim("F=ma and v=d/t")
        ["F=ma", "v=d/t"]
        >>> split_compound_claim("The limit exists. It equals zero.")
        ["The limit exists", "It equals zero"]
    """
    if not text or len(text.strip()) < 10:
        return [text.strip()] if text.strip() else []
    
    text = text.strip()
    atomic_claims = []
    
    # Step 1: Split by sentence boundaries (period, exclamation, question mark)
    sentences = re.split(r'([.!?]+)', text)
    # Reconstruct sentences with their punctuation
    current_sentence = ""
    for i, part in enumerate(sentences):
        if re.match(r'^[.!?]+$', part):
            current_sentence += part
            atomic_claims.append(current_sentence.strip())
            current_sentence = ""
        else:
            current_sentence += part
    if current_sentence.strip():
        atomic_claims.append(current_sentence.strip())
    
    # If we already have multiple sentences, return them
    if len(atomic_claims) > 1:
        atomic_claims = [c for c in atomic_claims if c and not re.match(r'^[.!?]+$', c)]
        return atomic_claims[:max_splits]
    
    # Step 2: Split by semicolons
    if ';' in text:
        parts = text.split(';')
        atomic_claims = [p.strip() for p in parts if p.strip()]
        if len(atomic_claims) > 1:
            return atomic_claims[:max_splits]
    
    # Step 3: Split by coordinating conjunctions
    # This is tricky because we need to avoid splitting noun phrases like "apples and oranges"
    # Heuristic: split if both sides have a verb (contains "is", "are", "equals", "=", etc.)
    conjunction_pattern = r'\s+\b(and|or|but)\b\s+'
    parts = re.split(conjunction_pattern, text, flags=re.IGNORECASE)
    
    if len(parts) > 1:
        # parts will be: [text1, "and", text2, "or", text3]
        # Reconstruct: text1, text2, text3
        atomic_claims = []
        for i, part in enumerate(parts):
            if part.lower() not in ['and', 'or', 'but']:
                if part.strip():
                    atomic_claims.append(part.strip())
        
        if len(atomic_claims) > 1:
            # Simple verb check: does each part look like a complete clause?
            # If all parts have verb-like patterns, return them
            verb_pattern = r'\b(is|are|was|were|equals|=|has|have|can|will|defined)\b'
            has_verbs = [bool(re.search(verb_pattern, claim, re.IGNORECASE)) for claim in atomic_claims]
            
            if sum(has_verbs) >= len(atomic_claims) / 2:  # At least half have verbs
                return atomic_claims[:max_splits]
    
    # If no splitting worked, return original as single claim
    return [text]


def enforce_granularity(
    claims: List[LearningClaim],
    max_propositions: int = 1
) -> List[LearningClaim]:
    """
    Enforce claim granularity policy by splitting compound claims.
    
    For each claim:
    1. Check if compound using is_compound_claim()
    2. If compound, split using split_compound_claim()
    3. Create new atomic claims preserving metadata
    4. Mark original claim as parent in metadata
    
    Args:
        claims: List of learning claims (potentially compound)
        max_propositions: Maximum propositions per claim (default 1 = atomic)
    
    Returns:
        List of atomic learning claims
    
    Side effects:
        - Logs INFO for each split operation
        - Preserves parent claim_id in metadata["parent_claim_id"]
    """
    if max_propositions != 1:
        logger.warning(
            f"max_propositions={max_propositions} != 1. "
            f"Current implementation only supports atomic claims (max_propositions=1)"
        )
    
    atomic_claims = []
    split_count = 0
    
    for claim in claims:
        # Skip claims with no text (e.g., evidence-first mode before generation)
        if not claim.claim_text:
            atomic_claims.append(claim)
            continue
        
        # Check if compound
        if not is_compound_claim(claim.claim_text):
            atomic_claims.append(claim)
            continue
        
        # Split compound claim
        logger.info(f"Splitting compound claim: {claim.claim_id[:8]}... '{claim.claim_text[:60]}...'")
        sub_texts = split_compound_claim(claim.claim_text)
        
        if len(sub_texts) <= 1:
            # Splitting didn't work, keep original
            atomic_claims.append(claim)
            continue
        
        split_count += 1
        
        # Create atomic sub-claims
        for i, sub_text in enumerate(sub_texts):
            # Create new claim copying metadata
            atomic_claim = LearningClaim(
                claim_type=claim.claim_type,
                claim_text=sub_text,
                evidence_ids=claim.evidence_ids.copy(),
                evidence_objects=claim.evidence_objects.copy(),
                confidence=claim.confidence,
                status=claim.status,
                rejection_reason=claim.rejection_reason,
                dependency_requests=claim.dependency_requests.copy(),
                metadata={
                    **claim.metadata,
                    "parent_claim_id": claim.claim_id,
                    "split_index": i,
                    "split_total": len(sub_texts)
                },
                created_at=claim.created_at
            )
            atomic_claims.append(atomic_claim)
        
        logger.info(f"Split into {len(sub_texts)} atomic claims")
    
    logger.info(f"Granularity enforcement: {split_count} compound claims split into {len(atomic_claims)} atomic claims")
    
    return atomic_claims
