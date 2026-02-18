"""
CS-Aware Verification Signals

Implements verification scoring based on computer science-specific features:
- Numeric consistency between claim and evidence
- Complexity notation consistency
- Code pattern anchoring
- Negation mismatch detection
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.verification.cs_claim_features import (
    CSClaimFeatureExtractor,
    NumericToken,
    ComplexityToken,
    CodeToken,
)

logger = logging.getLogger(__name__)


@dataclass
class VerificationSignal:
    """Result of a verification signal check."""
    signal_type: str  # "numeric", "complexity", "code", "negation"
    score: float  # 0.0-1.0
    evidence: str  # Description of what was verified
    has_match: bool  # True if claim/evidence match
    claim_features: Dict  # Features extracted from claim
    evidence_features: Dict  # Features extracted from evidence


class CSVerifier:
    """Verify CS-specific claims against evidence."""
    
    @staticmethod
    def numeric_consistency(claim_text: str, evidence_text: str) -> VerificationSignal:
        """
        Check numeric consistency between claim and evidence.
        
        If claim mentions specific numbers, evidence should reference same or related numbers.
        
        Args:
            claim_text: The claim statement
            evidence_text: The supporting evidence text
        
        Returns:
            VerificationSignal with consistency score
        """
        claim_numerics = CSClaimFeatureExtractor.extract_numeric_tokens(claim_text)
        evidence_numerics = CSClaimFeatureExtractor.extract_numeric_tokens(evidence_text)
        
        if not claim_numerics:
            # No numeric claim, so no mismatch
            return VerificationSignal(
                signal_type="numeric",
                score=1.0,
                evidence="No numeric tokens in claim",
                has_match=True,
                claim_features={'numeric_tokens': []},
                evidence_features={'numeric_tokens': []},
            )
        
        # Check if any claim numeric values appear in evidence
        claim_values = set(t.value for t in claim_numerics)
        evidence_values = set(t.value for t in evidence_numerics)
        
        # Look for exact matches or close matches
        matches = 0
        for cv in claim_values:
            for ev in evidence_values:
                if cv == ev:
                    matches += 1
                elif isinstance(cv, float) and isinstance(ev, float):
                    # Allow for small floating point variations
                    if abs(cv - ev) < 0.01 * max(abs(cv), abs(ev)):
                        matches += 1
        
        score = matches / len(claim_values) if claim_values else 1.0
        
        return VerificationSignal(
            signal_type="numeric",
            score=min(score, 1.0),
            evidence=f"Found {matches}/{len(claim_values)} matching numeric values",
            has_match=matches > 0 or len(claim_values) == 0,
            claim_features={'numeric_tokens': claim_numerics},
            evidence_features={'numeric_tokens': evidence_numerics},
        )
    
    @staticmethod
    def complexity_consistency(claim_text: str, evidence_text: str) -> VerificationSignal:
        """
        Check complexity notation consistency between claim and evidence.
        
        Big-O/Theta/Omega notations in claim should match or be related in evidence.
        
        Args:
            claim_text: The claim statement
            evidence_text: The supporting evidence text
        
        Returns:
            VerificationSignal with consistency score
        """
        claim_complexity = CSClaimFeatureExtractor.extract_complexity_tokens(claim_text)
        evidence_complexity = CSClaimFeatureExtractor.extract_complexity_tokens(evidence_text)
        
        if not claim_complexity:
            # No complexity claim
            return VerificationSignal(
                signal_type="complexity",
                score=1.0,
                evidence="No complexity tokens in claim",
                has_match=True,
                claim_features={'complexity_tokens': []},
                evidence_features={'complexity_tokens': []},
            )
        
        # Check for matching complexity expressions
        claim_expressions = set(ct.expression.lower() for ct in claim_complexity)
        evidence_expressions = set(ct.expression.lower() for ct in evidence_complexity)
        
        matches = 0
        for ce in claim_expressions:
            for ee in evidence_expressions:
                # Exact match or close match (ignoring whitespace differences)
                if ce.replace(' ', '') == ee.replace(' ', ''):
                    matches += 1
        
        score = matches / len(claim_expressions) if claim_expressions else 1.0
        
        return VerificationSignal(
            signal_type="complexity",
            score=min(score, 1.0),
            evidence=f"Found {matches}/{len(claim_expressions)} matching complexity expressions",
            has_match=matches > 0 or len(claim_expressions) == 0,
            claim_features={'complexity_tokens': claim_complexity},
            evidence_features={'complexity_tokens': evidence_complexity},
        )
    
    @staticmethod
    def code_anchor_score(claim_text: str, evidence_text: str) -> VerificationSignal:
        """
        Check if code-related claims have anchoring terms in evidence.
        
        Verifies that evidence contains supporting code patterns/concepts mentioned in claim.
        
        Args:
            claim_text: The claim statement
            evidence_text: The supporting evidence text
        
        Returns:
            VerificationSignal with anchor score
        """
        claim_code_tokens = CSClaimFeatureExtractor.extract_code_tokens(claim_text)
        evidence_code_tokens = CSClaimFeatureExtractor.extract_code_tokens(evidence_text)
        
        if not claim_code_tokens:
            # No code claim
            return VerificationSignal(
                signal_type="code",
                score=1.0,
                evidence="No code tokens in claim",
                has_match=True,
                claim_features={'code_tokens': []},
                evidence_features={'code_tokens': []},
            )
        
        # Check for matching code categories
        claim_categories = set(ct.category for ct in claim_code_tokens)
        evidence_categories = set(ct.category for ct in evidence_code_tokens)
        
        # Score based on category overlap
        matching_categories = claim_categories.intersection(evidence_categories)
        score = len(matching_categories) / len(claim_categories) if claim_categories else 1.0
        
        # Also check for specific patterns
        claim_patterns = set(ct.pattern.lower() for ct in claim_code_tokens)
        evidence_patterns = set(ct.pattern.lower() for ct in evidence_code_tokens)
        pattern_matches = len(claim_patterns.intersection(evidence_patterns))
        
        # Combine scores (weighted toward category matching)
        final_score = (0.6 * score) + (0.4 * (pattern_matches / len(claim_patterns) if claim_patterns else 0.0))
        
        return VerificationSignal(
            signal_type="code",
            score=min(final_score, 1.0),
            evidence=f"Matched {len(matching_categories)} code categories, {pattern_matches} patterns",
            has_match=len(matching_categories) > 0,
            claim_features={'code_tokens': claim_code_tokens},
            evidence_features={'code_tokens': evidence_code_tokens},
        )
    
    @staticmethod
    def negation_mismatch_penalty(claim_text: str, evidence_text: str) -> Tuple[float, str]:
        """
        Detect negation mismatch between claim and evidence.
        
        If claim asserts something is true but evidence uses negation (or vice versa),
        apply a penalty.
        
        Args:
            claim_text: The claim statement
            evidence_text: The supporting evidence text
        
        Returns:
            Tuple of (penalty_score, reason)
            penalty_score: 0.0 (no penalty) to -0.5 (strong mismatch)
            reason: Description of mismatch
        """
        claim_negation = CSClaimFeatureExtractor.detect_negation(claim_text)
        evidence_negation = CSClaimFeatureExtractor.detect_negation(evidence_text)
        
        # Penalize if negation presence differs significantly
        if claim_negation != evidence_negation:
            claim_count = CSClaimFeatureExtractor.count_negations(claim_text)
            evidence_count = CSClaimFeatureExtractor.count_negations(evidence_text)
            
            # Stronger penalty for strong mismatch
            severity = abs(claim_count - evidence_count) / max(1, max(claim_count, evidence_count))
            penalty = -0.5 * severity
            
            reason = f"Negation mismatch: claim has {claim_count}, evidence has {evidence_count}"
            return penalty, reason
        
        # No penalty if both affirm or both negate
        return 0.0, "Negation consistent between claim and evidence"
    
    @staticmethod
    def check_anchor_terms(
        claim_text: str,
        evidence_text: str,
        anchor_type: str
    ) -> Tuple[float, List[str]]:
        """
        Check if evidence contains anchor terms appropriate for claim type.
        
        Args:
            claim_text: The claim statement
            evidence_text: The supporting evidence text
            anchor_type: "complexity", "definition", or "code"
        
        Returns:
            Tuple of (score, found_anchors)
            score: 0.0-1.0 based on anchor term coverage
            found_anchors: List of anchor terms found in evidence
        """
        anchor_terms = CSClaimFeatureExtractor.find_anchor_terms(evidence_text, anchor_type)
        
        # Score based on anchor presence
        if anchor_type in ['complexity', 'definition', 'code']:
            # Require at least 1 anchor term for strong evidence
            score = 1.0 if anchor_terms else 0.5
        else:
            score = 0.5
        
        return score, anchor_terms


# Convenience wrapper functions
def numeric_consistency(claim: str, evidence: str) -> VerificationSignal:
    """Check numeric consistency between claim and evidence."""
    return CSVerifier.numeric_consistency(claim, evidence)


def complexity_consistency(claim: str, evidence: str) -> VerificationSignal:
    """Check complexity notation consistency."""
    return CSVerifier.complexity_consistency(claim, evidence)


def code_anchor_score(claim: str, evidence: str) -> VerificationSignal:
    """Check code pattern anchoring."""
    return CSVerifier.code_anchor_score(claim, evidence)


def negation_mismatch_penalty(claim: str, evidence: str) -> Tuple[float, str]:
    """Detect negation mismatch between claim and evidence."""
    return CSVerifier.negation_mismatch_penalty(claim, evidence)


def check_anchor_terms(claim: str, evidence: str, anchor_type: str) -> Tuple[float, List[str]]:
    """Check for anchor terms in evidence."""
    return CSVerifier.check_anchor_terms(claim, evidence, anchor_type)
