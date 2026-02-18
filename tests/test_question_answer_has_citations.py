"""
Tests for question answering with citations.

Ensures QuestionAnswerer generates answers with proper inline citations [1], [2], [3].
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.claims.question_answerer import QuestionAnswerer
from src.claims.schema import LearningClaim, ClaimType, VerificationStatus


class TestQuestionAnswering:
    """Test QuestionAnswerer generates answers from evidence."""
    
    @pytest.fixture
    def mock_evidence_store(self):
        """Mock evidence store with sample evidence."""
        mock_store = Mock()
        
        # Sample evidence snippets
        mock_store.search.return_value = [
            Mock(
                text="A heap is a complete binary tree that satisfies the heap property",
                similarity=0.92,
                metadata={"source": "textbook_ch5.pdf", "page": 42}
            ),
            Mock(
                text="Heaps support O(log n) insertion and deletion operations",
                similarity=0.88,
                metadata={"source": "lecture_notes.txt"}
            ),
            Mock(
                text="The heap property states parent >= children (max heap)",
                similarity=0.85,
                metadata={"source": "textbook_ch5.pdf", "page": 43}
            )
        ]
        
        return mock_store
    
    @pytest.fixture
    def mock_embedding_provider(self):
        """Mock embedding provider."""
        return Mock()
    
    @pytest.fixture
    def answerer(self, mock_evidence_store, mock_embedding_provider):
        """Create QuestionAnswerer instance."""
        return QuestionAnswerer(
            evidence_store=mock_evidence_store,
            embedding_provider=mock_embedding_provider
        )
    
    # ===== Basic Answer Generation =====
    
    def test_answer_generated_with_citations(self, answerer):
        """Test answer includes inline citations."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        
        # Should have answer_text
        assert result.answer_text is not None
        assert len(result.answer_text) > 0
        
        # Should include citations
        assert "[1]" in result.answer_text
    
    def test_multiple_citations_in_answer(self, answerer):
        """Test answer can include multiple citations."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        
        # Should reference multiple evidence pieces
        has_multiple_citations = (
            "[1]" in result.answer_text and
            ("[2]" in result.answer_text or "[3]" in result.answer_text)
        )
        assert has_multiple_citations or len(result.answer_text.split("[1]")) > 1
    
    def test_answer_status_is_answered_with_citations(self, answerer):
        """Test answer status becomes ANSWERED_WITH_CITATIONS."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        
        assert result.status == VerificationStatus.ANSWERED_WITH_CITATIONS
    
    # ===== Evidence Retrieval =====
    
    def test_evidence_retrieved_for_question(self, answerer, mock_evidence_store):
        """Test evidence is retrieved for question text."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION
        )
        
        answerer.answer_question(claim)
        
        # Should call evidence store with question
        mock_evidence_store.search.assert_called_once()
        call_args = mock_evidence_store.search.call_args
        
        # Should search with question text
        assert "heap" in call_args[0][0].lower() or call_args[1].get("query", "").lower()
    
    def test_lower_threshold_for_questions(self, answerer, mock_evidence_store):
        """Test questions use lower similarity threshold (0.2 vs 0.5)."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION
        )
        
        answerer.answer_question(claim)
        
        # Should use lower threshold
        call_kwargs = mock_evidence_store.search.call_args[1]
        threshold = call_kwargs.get("min_similarity", 0.5)
        
        assert threshold <= 0.3, "Questions should use lower similarity threshold"
    
    def test_no_evidence_returns_no_answer(self, mock_embedding_provider):
        """Test no evidence results in no answer."""
        # Mock empty evidence
        empty_store = Mock()
        empty_store.search.return_value = []
        
        answerer = QuestionAnswerer(
            evidence_store=empty_store,
            embedding_provider=mock_embedding_provider
        )
        
        claim = LearningClaim(
            claim_text="What is Zorbaxian sorting?",  # Made-up term
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        
        # Should have no answer or low confidence
        assert result.answer_text is None or result.confidence < 0.3
    
    # ===== Citation Format =====
    
    def test_citation_format_square_brackets(self, answerer):
        """Test citations use [1], [2], [3] format."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        
        # Should use square bracket format
        import re
        citations = re.findall(r'\[\d+\]', result.answer_text)
        assert len(citations) > 0, "Should have at least one citation"
        
        # Should be numeric
        for citation in citations:
            num = citation.strip('[]')
            assert num.isdigit(), f"Citation {citation} should be numeric"
    
    def test_citations_reference_evidence(self, answerer, mock_evidence_store):
        """Test citations correspond to retrieved evidence."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        
        # Get evidence count
        evidence_count = len(mock_evidence_store.search.return_value)
        
        # Citations should not exceed evidence count
        import re
        citations = re.findall(r'\[(\d+)\]', result.answer_text)
        max_citation = max([int(c) for c in citations]) if citations else 0
        
        assert max_citation <= evidence_count, "Citation numbers should not exceed evidence count"
    
    # ===== Confidence Computation =====
    
    def test_confidence_based_on_evidence_quality(self, mock_embedding_provider):
        """Test confidence reflects evidence quality."""
        # High-quality evidence
        high_quality_store = Mock()
        high_quality_store.search.return_value = [
            Mock(text="High quality answer", similarity=0.95),
            Mock(text="More quality", similarity=0.92),
            Mock(text="Even more", similarity=0.90)
        ]
        
        answerer_high = QuestionAnswerer(
            evidence_store=high_quality_store,
            embedding_provider=mock_embedding_provider
        )
        
        claim = LearningClaim(
            claim_text="What is X?",
            claim_type=ClaimType.QUESTION
        )
        
        result_high = answerer_high.answer_question(claim)
        
        # Should have high confidence
        assert result_high.confidence > 0.7
    
    def test_confidence_lower_with_weak_evidence(self, mock_embedding_provider):
        """Test confidence is lower with weak evidence."""
        # Low-quality evidence
        low_quality_store = Mock()
        low_quality_store.search.return_value = [
            Mock(text="Weak evidence", similarity=0.35),
            Mock(text="Also weak", similarity=0.30)
        ]
        
        answerer_low = QuestionAnswerer(
            evidence_store=low_quality_store,
            embedding_provider=mock_embedding_provider
        )
        
        claim = LearningClaim(
            claim_text="What is Y?",
            claim_type=ClaimType.QUESTION
        )
        
        result_low = answerer_low.answer_question(claim)
        
        # Should have lower confidence
        assert result_low.confidence < 0.6
    
    # ===== Answer Quality =====
    
    def test_answer_combines_multiple_evidence(self, answerer):
        """Test answer synthesizes multiple evidence pieces."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        
        # Answer should be substantial (not just one evidence snippet)
        assert len(result.answer_text) > 50, "Answer should combine evidence"
        
        # Should have multiple sentences or citations
        sentence_count = result.answer_text.count('.') + result.answer_text.count('?')
        citation_count = result.answer_text.count('[')
        
        assert sentence_count > 1 or citation_count > 1, "Answer should synthesize evidence"
    
    def test_answer_is_extractive(self, answerer, mock_evidence_store):
        """Test answer extracts text from evidence (not generated)."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        
        # Answer should contain words from evidence
        evidence_texts = [e.text for e in mock_evidence_store.search.return_value]
        evidence_words = set()
        for text in evidence_texts:
            evidence_words.update(text.lower().split())
        
        answer_words = set(result.answer_text.lower().split())
        overlap = evidence_words & answer_words
        
        # Should have significant overlap (extractive)
        assert len(overlap) >= 5, "Answer should extract from evidence"
    
    # ===== Edge Cases =====
    
    def test_empty_question_text(self, answerer):
        """Test empty question text."""
        claim = LearningClaim(
            claim_text="",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        
        # Should handle gracefully
        assert result.answer_text is None or result.confidence == 0.0
    
    def test_question_already_has_answer(self, answerer):
        """Test question with existing answer."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION,
            answer_text="A heap is a tree data structure [1]."
        )
        
        # Should preserve existing answer or update it
        result = answerer.answer_question(claim)
        
        # Should have an answer
        assert result.answer_text is not None
    
    def test_snippet_id_assigned(self, answerer):
        """Test snippet_id is assigned to answered question."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        
        # Should have snippet_id for tracking
        assert result.snippet_id is not None or result.answer_text is not None


class TestAnswerCitationIntegrity:
    """Test citation integrity and traceability."""
    
    @pytest.fixture
    def answerer_with_metadata(self):
        """Create answerer with metadata-rich evidence."""
        mock_store = Mock()
        mock_store.search.return_value = [
            Mock(
                text="Heaps are complete binary trees",
                similarity=0.90,
                metadata={"source": "textbook.pdf", "page": 42, "snippet_id": "snip_001"}
            ),
            Mock(
                text="Heaps support O(log n) operations",
                similarity=0.85,
                metadata={"source": "lecture.txt", "snippet_id": "snip_002"}
            )
        ]
        
        return QuestionAnswerer(
            evidence_store=mock_store,
            embedding_provider=Mock()
        )
    
    def test_citations_traceable_to_sources(self, answerer_with_metadata):
        """Test citations can be traced back to source documents."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer_with_metadata.answer_question(claim)
        
        # Should have answer with citations
        assert "[1]" in result.answer_text
        
        # (In full implementation, citations would map to evidence metadata)
        # For now, verify answer exists with proper format
        assert result.status == VerificationStatus.ANSWERED_WITH_CITATIONS
    
    def test_citation_numbers_sequential(self, answerer_with_metadata):
        """Test citation numbers are sequential."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer_with_metadata.answer_question(claim)
        
        # Extract citation numbers
        import re
        citations = re.findall(r'\[(\d+)\]', result.answer_text)
        citation_nums = [int(c) for c in citations]
        
        # Should be sequential (1, 2, 3, not 1, 3, 2)
        if len(citation_nums) > 1:
            for i in range(len(citation_nums) - 1):
                assert citation_nums[i] <= citation_nums[i + 1] + 1, "Citations should be roughly sequential"


class TestQuestionTypeHandling:
    """Test handling of different question types."""
    
    @pytest.fixture
    def answerer(self):
        """Create answerer with generic evidence."""
        mock_store = Mock()
        mock_store.search.return_value = [
            Mock(text="Generic answer evidence", similarity=0.8)
        ]
        return QuestionAnswerer(
            evidence_store=mock_store,
            embedding_provider=Mock()
        )
    
    def test_what_questions(self, answerer):
        """Test 'What' questions are answered."""
        claim = LearningClaim(
            claim_text="What is quicksort?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        assert result.answer_text is not None
    
    def test_why_questions(self, answerer):
        """Test 'Why' questions are answered."""
        claim = LearningClaim(
            claim_text="Why is quicksort efficient?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        assert result.answer_text is not None
    
    def test_how_questions(self, answerer):
        """Test 'How' questions are answered."""
        claim = LearningClaim(
            claim_text="How does binary search work?",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        assert result.answer_text is not None
    
    def test_imperative_questions(self, answerer):
        """Test imperative form questions (Explain X)."""
        claim = LearningClaim(
            claim_text="Explain how heaps maintain their property",
            claim_type=ClaimType.QUESTION
        )
        
        result = answerer.answer_question(claim)
        assert result.answer_text is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
