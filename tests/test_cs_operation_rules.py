"""
Tests for CS operation semantic rules checker.

Ensures CS-specific operation contradictions are detected:
- "Push removes" contradicts push semantics
- "Pop adds" contradicts pop semantics
- "Enqueue removes from front" contradicts enqueue semantics
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.claims.cs_operation_rules import (
    CSOperationRulesChecker,
    check_cs_operation_contradiction,
    OperationSemantics
)


class TestCSOperationRulesChecker:
    """Test CS operation semantic rules checker."""
    
    @pytest.fixture
    def checker(self):
        """Create CS rules checker instance."""
        return CSOperationRulesChecker(enabled=True)
    
    # ===== Stack Operations =====
    
    def test_push_removes_contradiction(self, checker):
        """Test that 'push removes' is detected as contradiction."""
        claim = "Stack push operation removes an element from the top"
        evidence = "The push operation adds elements to the stack"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        assert is_inconsistent, f"Expected contradiction, got: {reason}"
        assert "push" in reason.lower()
        assert "remove" in reason.lower() or "add" in reason.lower()
    
    def test_pop_adds_contradiction(self, checker):
        """Test that 'pop adds' is detected as contradiction."""
        claim = "Stack pop operation adds an element to the top"
        evidence = "Pop removes the top element from stack"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        assert is_inconsistent, f"Expected contradiction, got: {reason}"
        assert "pop" in reason.lower()
    
    def test_push_adds_correct(self, checker):
        """Test that 'push adds' is semantically correct."""
        claim = "Stack push operation adds an element to the top"
        evidence = "The push operation adds elements to the stack"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        assert not is_inconsistent, f"Unexpected contradiction detected: {reason}"
    
    def test_pop_removes_correct(self, checker):
        """Test that 'pop removes' is semantically correct."""
        claim = "Stack pop operation removes an element from the top"
        evidence = "Pop removes the top element from stack"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        assert not is_inconsistent, f"Unexpected contradiction detected: {reason}"
    
    # ===== Queue Operations =====
    
    def test_enqueue_removes_contradiction(self, checker):
        """Test that 'enqueue removes' is detected as contradiction."""
        claim = "Queue enqueue operation removes an element from the front"
        evidence = "Enqueue adds element to the rear of queue"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        assert is_inconsistent, f"Expected contradiction, got: {reason}"
        assert "enqueue" in reason.lower()
    
    def test_dequeue_adds_contradiction(self, checker):
        """Test that 'dequeue adds' is detected as contradiction."""
        claim = "Queue dequeue operation adds an element"
        evidence = "Dequeue removes element from front of queue"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        assert is_inconsistent, f"Expected contradiction, got: {reason}"
        assert "dequeue" in reason.lower()
    
    def test_enqueue_adds_correct(self, checker):
        """Test that 'enqueue adds' is semantically correct."""
        claim = "Queue enqueue operation adds an element to the rear"
        evidence = "Enqueue inserts element at rear of queue"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        assert not is_inconsistent, f"Unexpected contradiction detected: {reason}"
    
    def test_dequeue_removes_correct(self, checker):
        """Test that 'dequeue removes' is semantically correct."""
        claim = "Queue dequeue operation removes an element from the front"
        evidence = "Dequeue deletes element from front of queue"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        assert not is_inconsistent, f"Unexpected contradiction detected: {reason}"
    
    # ===== Heap Operations =====
    
    def test_insert_removes_contradiction(self, checker):
        """Test that 'insert removes' is detected as contradiction."""
        claim = "Heap insert operation removes an element"
        evidence = "Insert adds element to heap"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        assert is_inconsistent, f"Expected contradiction, got: {reason}"
        assert "insert" in reason.lower()
    
    def test_extract_min_adds_contradiction(self, checker):
        """Test that 'extract_min adds' is detected as contradiction."""
        claim = "Heap extract_min operation adds the minimum element"
        evidence = "Extract_min removes the minimum element from heap"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        assert is_inconsistent, f"Expected contradiction, got: {reason}"
        assert "extract" in reason.lower()
    
    # ===== Edge Cases =====
    
    def test_no_operation_in_claim(self, checker):
        """Test that claims without CS operations are not flagged."""
        claim = "Binary search has O(log n) time complexity"
        evidence = "Binary search runs in logarithmic time"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        assert not is_inconsistent, "Should not flag non-operation claims"
    
    def test_operation_name_only(self, checker):
        """Test that operation name alone doesn't trigger false positive."""
        claim = "The push operation is used in stacks"
        evidence = "Stacks use push and pop operations"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        # Should NOT be flagged (no explicit adds/removes contradiction)
        assert not is_inconsistent, f"Unexpected contradiction detected: {reason}"
    
    def test_disabled_checker(self):
        """Test that disabled checker returns no contradictions."""
        checker = CSOperationRulesChecker(enabled=False)
        
        claim = "Push removes elements from stack"
        evidence = "Push adds elements to stack"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        assert not is_inconsistent, "Disabled checker should not flag contradictions"
        assert reason is None
    
    def test_multiple_operations_in_claim(self, checker):
        """Test claim with multiple operations."""
        claim = "Push adds and pop removes elements from stack"
        evidence = "Stack operations: push adds to top, pop removes from top"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        # Should be consistent
        assert not is_inconsistent, f"Unexpected contradiction detected: {reason}"
    
    def test_contradictory_multiple_operations(self, checker):
        """Test claim with multiple operations, one contradictory."""
        claim = "Push adds and pop adds elements to stack"
        evidence = "Pop removes element from top of stack"
        
        is_inconsistent, reason = checker.check_claim_evidence_consistency(claim, evidence)
        
        # Should detect pop contradiction
        assert is_inconsistent, f"Expected contradiction, got: {reason}"
        assert "pop" in reason.lower()


class TestCSOperationConvenienceFunction:
    """Test convenience function for CS operation checking."""
    
    def test_convenience_function(self):
        """Test check_cs_operation_contradiction function."""
        claim = "Stack push operation removes an element"
        evidence = "Push adds element to stack"
        
        is_inconsistent, reason = check_cs_operation_contradiction(claim, evidence, enabled=True)
        
        assert is_inconsistent
        assert reason is not None
        assert "push" in reason.lower()
    
    def test_convenience_function_disabled(self):
        """Test convenience function with disabled flag."""
        claim = "Stack push operation removes an element"
        evidence = "Push adds element to stack"
        
        is_inconsistent, reason = check_cs_operation_contradiction(claim, evidence, enabled=False)
        
        assert not is_inconsistent
        assert reason is None


class TestOperationExtraction:
    """Test operation name extraction from text."""
    
    @pytest.fixture
    def checker(self):
        return CSOperationRulesChecker(enabled=True)
    
    def test_extract_push(self, checker):
        """Test extracting 'push' operation."""
        text = "The push operation adds elements"
        operations = checker._extract_operations(text)
        
        assert "push" in operations
    
    def test_extract_pop(self, checker):
        """Test extracting 'pop' operation."""
        text = "Stack pop removes top element"
        operations = checker._extract_operations(text)
        
        assert "pop" in operations
    
    def test_extract_multiple(self, checker):
        """Test extracting multiple operations."""
        text = "Stack uses push and pop operations"
        operations = checker._extract_operations(text)
        
        assert "push" in operations
        assert "pop" in operations
    
    def test_no_operations(self, checker):
        """Test text with no CS operations."""
        text = "Binary search is efficient for sorted arrays"
        operations = checker._extract_operations(text)
        
        assert len(operations) == 0


class TestTextImpliesActions:
    """Test action verb detection in text."""
    
    @pytest.fixture
    def checker(self):
        return CSOperationRulesChecker(enabled=True)
    
    def test_text_implies_adds(self, checker):
        """Test detection of 'adds' implication."""
        text = "push adds element to stack"
        implies_adds = checker._text_implies_adds(text, "push")
        
        assert implies_adds
    
    def test_text_implies_removes(self, checker):
        """Test detection of 'removes' implication."""
        text = "pop removes element from stack"
        implies_removes = checker._text_implies_removes(text, "pop")
        
        assert implies_removes
    
    def test_no_implication_adds(self, checker):
        """Test text without 'adds' implication."""
        text = "push is a stack operation"
        implies_adds = checker._text_implies_adds(text, "push")
        
        assert not implies_adds
    
    def test_no_implication_removes(self, checker):
        """Test text without 'removes' implication."""
        text = "pop is a stack operation"
        implies_removes = checker._text_implies_removes(text, "pop")
        
        assert not implies_removes


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v", "-s"])
