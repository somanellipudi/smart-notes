"""
CS Operation Semantic Rules Checker.

Detects semantic contradictions in CS data structure operation claims.
Example: "Push removes element from stack" contradicts known push semantics.

Usage:
    checker = CSOperationRulesChecker()
    is_inconsistent = checker.check_claim_evidence_consistency(
        claim="Push removes element from stack",
        evidence="Push adds element to top of stack"
    )
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OperationSemantics:
    """Semantic expectations for CS operations."""
    operation: str
    adds: bool  # True if operation adds elements
    removes: bool  # True if operation removes elements
    structure: str  # stack, queue, heap, etc.
    end: Optional[str] = None  # top, bottom, front, rear


class CSOperationRulesChecker:
    """
    Check CS operation semantic consistency between claims and evidence.
    
    Detects contradictions like:
    - "Push removes element" (push should add)
    - "Pop adds element" (pop should remove)
    - "Enqueue removes from front" (enqueue should add to rear)
    """
    
    # Operation semantic rules
    OPERATION_RULES = {
        "push": OperationSemantics(
            operation="push",
            adds=True,
            removes=False,
            structure="stack",
            end="top"
        ),
        "pop": OperationSemantics(
            operation="pop",
            adds=False,
            removes=True,
            structure="stack",
            end="top"
        ),
        "enqueue": OperationSemantics(
            operation="enqueue",
            adds=True,
            removes=False,
            structure="queue",
            end="rear"
        ),
        "dequeue": OperationSemantics(
            operation="dequeue",
            adds=False,
            removes=True,
            structure="queue",
            end="front"
        ),
        "insert": OperationSemantics(
            operation="insert",
            adds=True,
            removes=False,
            structure="heap"
        ),
        "extract_min": OperationSemantics(
            operation="extract_min",
            adds=False,
            removes=True,
            structure="heap"
        ),
        "extract_max": OperationSemantics(
            operation="extract_max",
            adds=False,
            removes=True,
            structure="heap"
        ),
    }
    
    # Action verbs for adds/removes detection
    ADD_VERBS = {
        "add", "adds", "added", "adding",
        "insert", "inserts", "inserted", "inserting",
        "place", "places", "placed", "placing",
        "put", "puts"
    }
    
    REMOVE_VERBS = {
        "remove", "removes", "removed", "removing",
        "delete", "deletes", "deleted", "deleting",
        "extract", "extracts", "extracted", "extracting",
        "take", "takes", "took", "taking"
    }
    
    def __init__(self, enabled: bool = True):
        """
        Initialize CS operation rules checker.
        
        Args:
            enabled: Whether to enable checks
        """
        self.enabled = enabled
    
    def check_claim_evidence_consistency(
        self,
        claim: str,
        evidence: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if claim and evidence have CS operation semantic contradictions.
        
        Args:
            claim: Claim text
            evidence: Evidence text
        
        Returns:
            Tuple of (is_inconsistent, reason)
            - is_inconsistent: True if semantic contradiction detected
            - reason: Human-readable explanation (if inconsistent)
        """
        if not self.enabled:
            return False, None
        
        # Detect operations in claim
        claim_ops = self._extract_operations(claim)
        if not claim_ops:
            return False, None  # No CS operations detected
        
        # Check each operation in claim
        for op_name in claim_ops:
            if op_name not in self.OPERATION_RULES:
                continue
            
            op_semantics = self.OPERATION_RULES[op_name]
            
            # Check if claim contradicts operation semantics
            claim_contradiction = self._check_text_contradicts_operation(
                claim, op_semantics
            )
            
            if claim_contradiction:
                reason = claim_contradiction
                logger.debug(f"CS operation contradiction detected: {reason}")
                return True, reason
            
            # Check if evidence contradicts operation semantics
            evidence_contradiction = self._check_text_contradicts_operation(
                evidence, op_semantics
            )
            
            if evidence_contradiction:
                reason = evidence_contradiction
                logger.debug(f"CS operation contradiction in evidence: {reason}")
                return True, reason
        
        return False, None
    
    def _extract_operations(self, text: str) -> List[str]:
        """Extract CS operation names from text."""
        text_lower = text.lower()
        operations = []
        
        for op_name in self.OPERATION_RULES.keys():
            # Match whole word operation names
            pattern = r'\b' + re.escape(op_name) + r'\b'
            if re.search(pattern, text_lower):
                operations.append(op_name)
        
        return operations
    
    def _check_text_contradicts_operation(
        self,
        text: str,
        op_semantics: OperationSemantics
    ) -> Optional[str]:
        """
        Check if text contradicts operation semantics.
        
        Returns:
            Contradiction reason (if found), else None
        """
        text_lower = text.lower()
        op_name = op_semantics.operation
        
        # Split text on conjunctions to handle "push adds and pop removes" correctly
        # This prevents matching across different clauses
        text_segments = re.split(r'\b(?:and|but|while|whereas)\b', text_lower)
        
        # Check each segment for contradictions
        for segment in text_segments:
            # Skip segments that don't mention the operation
            if op_name not in segment:
                continue
            
            # Check if text says operation adds elements when it shouldn't
            if not op_semantics.adds:
                if self._text_implies_adds(segment, op_name):
                    return (
                        f"'{op_name}' should not add elements, "
                        f"but text implies it adds"
                    )
            
            # Check if text says operation removes elements when it shouldn't
            if not op_semantics.removes:
                if self._text_implies_removes(segment, op_name):
                    return (
                        f"'{op_name}' should not remove elements, "
                        f"but text implies it removes"
                    )
            
            # Check if text says operation adds when it should remove
            if op_semantics.removes and not op_semantics.adds:
                if self._text_implies_adds(segment, op_name):
                    return (
                        f"'{op_name}' should remove elements, "
                        f"but text implies it adds"
                    )
            
            # Check if text says operation removes when it should add
            if op_semantics.adds and not op_semantics.removes:
                if self._text_implies_removes(segment, op_name):
                    return (
                        f"'{op_name}' should add elements, "
                        f"but text implies it removes"
                    )
        
        return None
    
    def _text_implies_adds(self, text: str, operation: str) -> bool:
        """Check if text implies operation adds elements."""
        # Look for patterns like "push adds", "push inserts"
        for add_verb in self.ADD_VERBS:
            # Skip if verb == operation (e.g., "push pushes" is not informative)
            if add_verb == operation:
                continue
            
            # Pattern 1: operation + add_verb (e.g., "push adds")
            pattern = r'\b' + re.escape(operation) + r'\s+' + re.escape(add_verb) + r'\b'
            if re.search(pattern, text):
                return True
            
            # Pattern 2: operation + "operation" + add_verb (e.g., "push operation adds")
            pattern = (
                r'\b' + re.escape(operation) + 
                r'\s+operation\s+' + 
                re.escape(add_verb) + r'\b'
            )
            if re.search(pattern, text):
                return True
            
            # Pattern 3: operation + 1-3 words + add_verb
            # But exclude phrases with "is used"
            pattern = (
                r'\b' + re.escape(operation) + 
                r'(?:(?!\bis\s+used\b)\s+\w+){0,3}?\s+' + 
                re.escape(add_verb) + r'\b'
            )
            if re.search(pattern, text):
                return True
        
        return False
    
    def _text_implies_removes(self, text: str, operation: str) -> bool:
        """Check if text implies operation removes elements."""
        # Look for patterns like "pop removes", "pop deletes"
        for remove_verb in self.REMOVE_VERBS:
            # Skip if verb == operation
            if remove_verb == operation:
                continue
            
            # Pattern 1: operation + remove_verb (e.g., "pop removes")
            pattern = r'\b' + re.escape(operation) + r'\s+' + re.escape(remove_verb) + r'\b'
            if re.search(pattern, text):
                return True
            
            # Pattern 2: operation + "operation" + remove_verb (e.g., "pop operation removes")
            pattern = (
                r'\b' + re.escape(operation) + 
                r'\s+operation\s+' + 
                re.escape(remove_verb) + r'\b'
            )
            if re.search(pattern, text):
                return True
            
            # Pattern 3: operation + 1-3 words + remove_verb
            # But exclude phrases with "is used"
            pattern = (
                r'\b' + re.escape(operation) + 
                r'(?:(?!\bis\s+used\b)\s+\w+){0,3}?\s+' + 
                re.escape(remove_verb) + r'\b'
            )
            if re.search(pattern, text):
                return True
        
        return False


def check_cs_operation_contradiction(
    claim: str,
    evidence: str,
    enabled: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to check CS operation contradiction.
    
    Args:
        claim: Claim text
        evidence: Evidence text
        enabled: Whether to enable checks
    
    Returns:
        Tuple of (is_inconsistent, reason)
    """
    checker = CSOperationRulesChecker(enabled=enabled)
    return checker.check_claim_evidence_consistency(claim, evidence)
