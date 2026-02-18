"""
Question and Claim Type Detector.

Automatically detects claim types from text:
- QUESTION: Study questions requiring answers
- MISCONCEPTION: False statements to be corrected
- FACT_CLAIM: Verifiable factual assertions (default)

This prevents questions from going through claim verification pipeline.
"""

import re
import logging
from typing import Tuple, Optional
from .schema import ClaimType

logger = logging.getLogger(__name__)


class ClaimTypeDetector:
    """
    Detect claim type from text content.
    
    Rules:
    1. QUESTION if:
       - Text ends with '?'
       - Text starts with question words (What, Why, How, When, Where, Who, Which)
       - Text contains "Explain", "Describe" followed by question patterns
    
    2. MISCONCEPTION if:
       - Text contains "misconception", "common mistake", "false belief"
       - Text contains "people think", "incorrectly believe", "wrong idea"
    
    3. FACT_CLAIM (default):
       - All other cases
    """
    
    # Question starters (case-insensitive)
    QUESTION_STARTERS = {
        "what", "why", "how", "when", "where", "who", "which", "whom",
        "explain", "describe", "define", "compare", "contrast",
        "list", "enumerate", "identify", "name", "state", "give"
    }
    
    # Misconception indicators (case-insensitive)
    MISCONCEPTION_INDICATORS = {
        "misconception", "common mistake", "false belief", "incorrect belief",
        "people think", "incorrectly believe", "wrong idea", "mistaken belief",
        "common error", "frequent error", "typical mistake", "often confused"
    }
    
    def __init__(self):
        """Initialize detector."""
        pass
    
    def detect(self, text: str, hint: Optional[str] = None) -> ClaimType:
        """
        Detect claim type from text.
        
        Args:
            text: Text content to analyze
            hint: Optional hint about expected type (e.g., from source context)
        
        Returns:
            ClaimType enum value
        """
        if not text or not text.strip():
            return ClaimType.FACT_CLAIM
        
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        # Check for question patterns
        if self._is_question(text_clean, text_lower):
            logger.debug(f"Detected QUESTION: '{text_clean[:50]}...'")
            return ClaimType.QUESTION
        
        # Check for misconception patterns
        if self._is_misconception(text_lower, hint):
            logger.debug(f"Detected MISCONCEPTION: '{text_clean[:50]}...'")
            return ClaimType.MISCONCEPTION
        
        # Default to fact claim
        return ClaimType.FACT_CLAIM
    
    def _is_question(self, text: str, text_lower: str) -> bool:
        """
        Check if text is a question.
        
        Args:
            text: Original text (preserves punctuation)
            text_lower: Lowercased text
        
        Returns:
            True if text is a question
        """
        # Rule 1: Ends with question mark
        if text.endswith('?'):
            return True
        
        # Rule 2: Starts with question word
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in self.QUESTION_STARTERS:
            # Additional validation: question words followed by reasonable patterns
            # Avoid false positives like "Define X as Y" (statement, not question)
            
            # If it has ?, definitely a question
            if '?' in text:
                return True
            
            # Check for "Define X as Y" pattern (statement, not question)
            if first_word == "define":
                # If has " as " pattern, it's a definition statement
                if " as " in text_lower or " is " in text_lower:
                    return False
            
            # Check for imperative commands (Explain, Describe, etc.)
            if first_word in {"explain", "describe", "define", "compare", "list", "identify"}:
                # If it's short and doesn't have punctuation suggesting statement, likely a question
                if len(text.split()) < 20 and not text.endswith('.'):
                    return True
                # If it has question-like structure
                if any(qw in text_lower for qw in ["what", "why", "how", "when", "where"]):
                    return True
            
            # Standard question words (What, Why, How, etc.)
            if first_word in {"what", "why", "how", "when", "where", "who", "which", "whom"}:
                return True
        
        # Rule 3: Contains question structure in middle
        # Pattern: "Keyword question_word ..." (e.g., "Explain how X works")
        question_pattern = r'\b(explain|describe|tell|show)\s+(how|why|what|when|where|which|who)\b'
        if re.search(question_pattern, text_lower):
            return True
        
        return False
    
    def _is_misconception(self, text_lower: str, hint: Optional[str] = None) -> bool:
        """
        Check if text describes a misconception.
        
        Args:
            text_lower: Lowercased text
            hint: Optional hint from source context
        
        Returns:
            True if text is a misconception
        """
        # Check hint first
        if hint and "misconception" in hint.lower():
            return True
        
        # Check text for misconception indicators
        for indicator in self.MISCONCEPTION_INDICATORS:
            if indicator in text_lower:
                return True
        
        return False
    
    def detect_with_metadata(self, text: str, metadata: dict = None) -> Tuple[ClaimType, dict]:
        """
        Detect claim type and return additional metadata.
        
        Args:
            text: Text content to analyze
            metadata: Optional existing metadata dict
        
        Returns:
            Tuple of (ClaimType, updated_metadata)
        """
        metadata = metadata or {}
        
        claim_type = self.detect(
            text,
            hint=metadata.get("source_section") or metadata.get("ui_display")
        )
        
        # Add detection metadata
        metadata["detected_type"] = claim_type.value
        metadata["is_question"] = (claim_type == ClaimType.QUESTION)
        metadata["is_misconception"] = (claim_type == ClaimType.MISCONCEPTION)
        
        if claim_type == ClaimType.QUESTION:
            metadata["requires_answer"] = True
            metadata["skip_verification"] = True  # Questions don't get verified
        
        return claim_type, metadata


# Global detector instance
_detector = ClaimTypeDetector()


def detect_claim_type(text: str, hint: Optional[str] = None) -> ClaimType:
    """
    Convenience function to detect claim type from text.
    
    Args:
        text: Text content to analyze
        hint: Optional hint about expected type
    
    Returns:
        ClaimType enum value
    
    Examples:
        >>> detect_claim_type("What is a stack?")
        ClaimType.QUESTION
        
        >>> detect_claim_type("How does quicksort work?")
        ClaimType.QUESTION
        
        >>> detect_claim_type("A common misconception is that quicksort is always O(n log n)")
        ClaimType.MISCONCEPTION
        
        >>> detect_claim_type("Quicksort has O(n log n) average time complexity")
        ClaimType.FACT_CLAIM
    """
    return _detector.detect(text, hint)


def is_question(text: str) -> bool:
    """
    Quick check if text is a question.
    
    Args:
        text: Text content
    
    Returns:
        True if text is detected as a question
    """
    return detect_claim_type(text) == ClaimType.QUESTION


def is_misconception(text: str, hint: Optional[str] = None) -> bool:
    """
    Quick check if text describes a misconception.
    
    Args:
        text: Text content
        hint: Optional hint from context
    
    Returns:
        True if text is detected as a misconception
    """
    return detect_claim_type(text, hint) == ClaimType.MISCONCEPTION
