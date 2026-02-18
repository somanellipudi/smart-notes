"""
Tests for question type detection.

Ensures questions are correctly identified and not treated as factual claims.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.claims.claim_type_detector import (
    ClaimTypeDetector,
    detect_claim_type,
    is_question,
    is_misconception
)
from src.claims.schema import ClaimType


class TestQuestionDetection:
    """Test question detection from various text patterns."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return ClaimTypeDetector()
    
    # ===== Basic Question Marks =====
    
    def test_ends_with_question_mark(self, detector):
        """Test that text ending with ? is detected as question."""
        text = "What is a stack?"
        result = detector.detect(text)
        assert result == ClaimType.QUESTION
    
    def test_multiple_questions(self, detector):
        """Test multiple question marks."""
        text = "What is recursion?"
        result = detector.detect(text)
        assert result == ClaimType.QUESTION
    
    # ===== Question Word Starters =====
    
    def test_what_question(self, detector):
        """Test 'What' questions."""
        texts = [
            "What is a binary search tree?",
            "What are the properties of a heap?",
            "What happens during a stack overflow?"
        ]
        for text in texts:
            result = detector.detect(text)
            assert result == ClaimType.QUESTION, f"Failed on: {text}"
    
    def test_why_question(self, detector):
        """Test 'Why' questions."""
        texts = [
            "Why is quicksort faster than bubble sort?",
            "Why does HashMap use O(1) lookup?",
            "Why is recursion useful?"
        ]
        for text in texts:
            result = detector.detect(text)
            assert result == ClaimType.QUESTION, f"Failed on: {text}"
    
    def test_how_question(self, detector):
        """Test 'How' questions."""
        texts = [
            "How does BFS work?",
            "How do you implement a linked list?",
            "How is memory allocated in Java?"
        ]
        for text in texts:
            result = detector.detect(text)
            assert result == ClaimType.QUESTION, f"Failed on: {text}"
    
    def test_when_where_who_questions(self, detector):
        """Test when/where/who questions."""
        texts = [
            "When should you use a stack?",
            "Where is the heap stored?",
            "Who invented the linked list?"
        ]
        for text in texts:
            result = detector.detect(text)
            assert result == ClaimType.QUESTION, f"Failed on: {text}"
    
    # ===== Imperative Question Forms =====
    
    def test_explain_questions(self, detector):
        """Test 'Explain' imperative questions."""
        texts = [
            "Explain how quicksort works",
            "Explain what recursion is",
            "Explain why we use heaps"
        ]
        for text in texts:
            result = detector.detect(text)
            assert result == ClaimType.QUESTION, f"Failed on: {text}"
    
    def test_describe_questions(self, detector):
        """Test 'Describe' imperative questions."""
        texts = [
            "Describe the BFS algorithm",
            "Describe how DFS differs from BFS"
        ]
        for text in texts:
            result = detector.detect(text)
            assert result == ClaimType.QUESTION, f"Failed on: {text}"
    
    # ===== Non-Questions (Should be FACT_CLAIM) =====
    
    def test_fact_statements(self, detector):
        """Test factual statements are not questions."""
        texts = [
            "Quicksort has O(n log n) average time complexity",
            "A stack is a LIFO data structure",
            "Binary search requires sorted input"
        ]
        for text in texts:
            result = detector.detect(text)
            assert result == ClaimType.FACT_CLAIM, f"Failed on: {text}"
    
    def test_definition_statements(self, detector):
        """Test 'Define X as Y' statements are not questions."""
        texts = [
            "Define recursion as a function calling itself",
            "A heap is defined as a complete binary tree"
        ]
        for text in texts:
            result = detector.detect(text)
            # These should NOT be detected as questions (statements, not questions)
            assert result == ClaimType.FACT_CLAIM, f"Failed on: {text}"
    
    # ===== Edge Cases =====
    
    def test_empty_text(self, detector):
        """Test empty text defaults to FACT_CLAIM."""
        result = detector.detect("")
        assert result == ClaimType.FACT_CLAIM
    
    def test_whitespace_only(self, detector):
        """Test whitespace-only text."""
        result = detector.detect("   \n\t  ")
        assert result == ClaimType.FACT_CLAIM
    
    def test_question_word_in_middle(self, detector):
        """Test question word not at start."""
        text = "The algorithm shows how to sort efficiently."
        result = detector.detect(text)
        # Should NOT be question (question word in middle of sentence)
        assert result == ClaimType.FACT_CLAIM


class TestMisconceptionDetection:
    """Test misconception detection."""
    
    @pytest.fixture
    def detector(self):
        return ClaimTypeDetector()
    
    def test_explicit_misconception(self, detector):
        """Test explicit 'misconception' keyword."""
        texts = [
            "A common misconception is that quicksort is always O(n log n)",
            "The misconception that stacks are faster than arrays",
            "Misconception: Python is interpreted at runtime"
        ]
        for text in texts:
            result = detector.detect(text)
            assert result == ClaimType.MISCONCEPTION, f"Failed on: {text}"
    
    def test_common_mistake(self, detector):
        """Test 'common mistake' indicator."""
        texts = [
            "A common mistake is forgetting base cases in recursion",
            "Common error: confusing BFS with DFS"
        ]
        for text in texts:
            result = detector.detect(text)
            assert result == ClaimType.MISCONCEPTION, f"Failed on: {text}"
    
    def test_people_think(self, detector):
        """Test 'people think' indicator."""
        texts = [
            "People think that hashmaps are always O(1)",
            "Many incorrectly believe that all sorting is O(n^2)"
        ]
        for text in texts:
            result = detector.detect(text)
            assert result == ClaimType.MISCONCEPTION, f"Failed on: {text}"
    
    def test_hint_parameter(self, detector):
        """Test hint parameter for misconception detection."""
        text = "Quicksort is always O(n log n)"
        result = detector.detect(text, hint="common_mistakes")
        # Without hint, might be FACT_CLAIM
        # With hint, should be MISCONCEPTION
        result_with_hint = detector.detect(text, hint="misconception: ...")
        assert result_with_hint == ClaimType.MISCONCEPTION


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_is_question_true(self):
        """Test is_question returns True for questions."""
        assert is_question("What is a stack?")
        assert is_question("How does BFS work?")
        assert is_question("Why use recursion?")
    
    def test_is_question_false(self):
        """Test is_question returns False for non-questions."""
        assert not is_question("A stack is a LIFO structure")
        assert not is_question("Binary search is O(log n)")
    
    def test_is_misconception_true(self):
        """Test is_misconception returns True."""
        assert is_misconception("A common misconception is...")
        assert is_misconception("People think that sorting is always slow")
    
    def test_is_misconception_false(self):
        """Test is_misconception returns False."""
        assert not is_misconception("Quicksort has O(n log n) complexity")
    
    def test_detect_claim_type_shortcut(self):
        """Test detect_claim_type convenience function."""
        assert detect_claim_type("What is BFS?") == ClaimType.QUESTION
        assert detect_claim_type("BFS is a graph algorithm") == ClaimType.FACT_CLAIM
        assert detect_claim_type("Misconception: BFS is always faster") == ClaimType.MISCONCEPTION


class TestAutoDetectionWithMetadata:
    """Test detect_with_metadata method."""
    
    @pytest.fixture
    def detector(self):
        return ClaimTypeDetector()
    
    def test_question_metadata(self, detector):
        """Test metadata for detected questions."""
        text = "What is a heap?"
        claim_type, metadata = detector.detect_with_metadata(text)
        
        assert claim_type == ClaimType.QUESTION
        assert metadata["detected_type"] == "question"
        assert metadata["is_question"] is True
        assert metadata["requires_answer"] is True
        assert metadata["skip_verification"] is True
    
    def test_fact_metadata(self, detector):
        """Test metadata for detected facts."""
        text = "A heap is a complete binary tree"
        claim_type, metadata = detector.detect_with_metadata(text)
        
        assert claim_type == ClaimType.FACT_CLAIM
        assert metadata["detected_type"] == "fact_claim"
        assert metadata["is_question"] is False
    
    def test_existing_metadata_preserved(self, detector):
        """Test existing metadata is preserved and extended."""
        text = "What is BFS?"
        existing = {"source": "faq", "index": 5}
        
        claim_type, metadata = detector.detect_with_metadata(text, existing)
        
        assert metadata["source"] == "faq"
        assert metadata["index"] == 5
        assert metadata["detected_type"] == "question"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
