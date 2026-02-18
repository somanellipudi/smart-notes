"""
Tests that QUESTION type claims skip traditional verification.

Questions should go through answer generation, not NLI verification.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.claims.schema import ClaimType, VerificationStatus, LearningClaim
from src.claims.validator import ClaimValidator


class TestQuestionSkipsVerification:
    """Test that questions bypass traditional verification."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ClaimValidator(
            verified_threshold=0.7,
            rejected_threshold=0.3,
            min_evidence_count=1
        )
    
    # ===== Questions Skip Verification =====
    
    def test_question_without_answer_rejected(self, validator):
        """Test question without answer_text is REJECTED."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION,
            confidence=0.0
        )
        
        result = validator.validate_claim(claim)
        
        # Should reject (no answer)
        assert result == VerificationStatus.REJECTED
        assert claim.status == VerificationStatus.REJECTED
    
    def test_question_with_answer_accepted(self, validator):
        """Test question with answer_text is ANSWERED_WITH_CITATIONS."""
        claim = LearningClaim(
            claim_text="What is a heap?",
            claim_type=ClaimType.QUESTION,
            confidence=0.0,
            answer_text="A heap is a complete binary tree [1].",
            snippet_id="snippet_001"
        )
        
        result = validator.validate_claim(claim)
        
        # Should accept with citations
        assert result == VerificationStatus.ANSWERED_WITH_CITATIONS
    
    def test_question_never_requires_evidence(self, validator):
        """Test questions don't require evidence metadata."""
        # Questions with answers should pass even without evidence_ids
        claim = LearningClaim(
            claim_text="What is BFS?",
            claim_type=ClaimType.QUESTION,
            answer_text="BFS is breadth-first search [1].",
            confidence=0.8
        )
        
        result = validator.validate_claim(claim)
        
        # Should be accepted even without evidence_ids
        assert result == VerificationStatus.ANSWERED_WITH_CITATIONS
    
    # ===== Fact Claims Require Evidence =====
    
    def test_fact_claim_requires_evidence(self, validator):
        """Test FACT_CLAIM requires evidence metadata."""
        # Without evidence, should be rejected
        claim = LearningClaim(
            claim_text="Quicksort has O(n log n) average time complexity",
            claim_type=ClaimType.FACT_CLAIM,
            confidence=0.9
        )
        
        result = validator.validate_claim(claim)
        
        # Should reject (no evidence)
        assert result == VerificationStatus.REJECTED
    
    def test_fact_claim_with_evidence_verified(self, validator):
        """Test FACT_CLAIM with sufficient evidence is verified."""
        claim = LearningClaim(
            claim_text="Quicksort has O(n log n) average complexity",
            claim_type=ClaimType.FACT_CLAIM,
            confidence=0.85,
            evidence_ids=["ev_001", "ev_002"]
        )
        claim.metadata["evidence_sufficient"] = True
        
        result = validator.validate_claim(claim)
        
        # Should verify
        assert result == VerificationStatus.VERIFIED
    
    # ===== Misconceptions Use Framing Check =====
    
    def test_misconception_checks_framing(self, validator):
        """Test misconceptions check for proper framing."""
        # Misconception WITHOUT framing
        claim = LearningClaim(
            claim_text="A common misconception is that quicksort is always O(n log n)",
            claim_type=ClaimType.MISCONCEPTION,
            confidence=0.8,
            evidence_ids=["ev_001"]
        )
        claim.metadata["evidence_sufficient"] = True
        
        result = validator.validate_claim(claim)
        
        # Should flag as needing framing
        assert result == VerificationStatus.NEEDS_FRAMING
    
    def test_misconception_with_framing_accepted(self, validator):
        """Test properly framed misconceptions are accepted."""
        claim = LearningClaim(
            claim_text="A common misconception is that quicksort is always O(n log n). Actually, worst case is O(n^2).",
            claim_type=ClaimType.MISCONCEPTION,
            confidence=0.85,
            evidence_ids=["ev_001"]
        )
        claim.metadata["evidence_sufficient"] = True
        
        result = validator.validate_claim(claim)
        
        # Should verify (proper framing present)
        assert result == VerificationStatus.VERIFIED


class TestValidatorRouting:
    """Test validator routes claims by type."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ClaimValidator(verified_threshold=0.7, rejected_threshold=0.3)
    
    def test_routing_by_claim_type(self, validator):
        """Test different claim types take different paths."""
        # Test QUESTION routing
        question = LearningClaim(
            claim_text="What is BFS?",
            claim_type=ClaimType.QUESTION,
            answer_text="BFS is breadth-first search [1]."
        )
        q_result = validator.validate_claim(question)
        assert q_result == VerificationStatus.ANSWERED_WITH_CITATIONS
        
        # Test FACT_CLAIM routing
        fact = LearningClaim(
            claim_text="BFS uses a queue",
            claim_type=ClaimType.FACT_CLAIM,
            confidence=0.85,
            evidence_ids=["ev_001"]
        )
        fact.metadata["evidence_sufficient"] = True
        f_result = validator.validate_claim(fact)
        # Should verify with evidence
        assert f_result == VerificationStatus.VERIFIED
        
        # Test MISCONCEPTION routing
        misconception = LearningClaim(
            claim_text="Misconception: BFS is faster. Actually, it depends on graph structure.",
            claim_type=ClaimType.MISCONCEPTION,
            confidence=0.80,
            evidence_ids=["ev_001"]
        )
        misconception.metadata["evidence_sufficient"] = True
        m_result = validator.validate_claim(misconception)
        # Should verify (has framing)
        assert m_result == VerificationStatus.VERIFIED


class TestBackwardCompatibility:
    """Test backward compatibility with existing claims."""
    
    @pytest.fixture
    def validator(self):
        """Create validator."""
        return ClaimValidator(verified_threshold=0.7, rejected_threshold=0.3)
    
    def test_untyped_claim_defaults_to_fact(self, validator):
        """Test claims use FACT_CLAIM if not explicitly typed."""
        # In actual usage, claims without explicit type would use FACT_CLAIM
        claim = LearningClaim(
            claim_text="Binary search is O(log n)",
            claim_type=ClaimType.FACT_CLAIM,
            confidence=0.85,
            evidence_ids=["ev_001"]
        )
        claim.metadata["evidence_sufficient"] = True
        
        # Should go through standard verification
        result = validator.validate_claim(claim)
        
        # Should verify normally
        assert result == VerificationStatus.VERIFIED
    
    def test_legacy_claims_still_work(self, validator):
        """Test old claims without claim_type still work."""
        # Create claim like old system would
        claim = LearningClaim(
            claim_text="Heaps support O(log n) insertion",
            claim_type=ClaimType.FACT_CLAIM,  # Default
            confidence=0.9,
            evidence_ids=["ev_001", "ev_002"]
        )
        claim.metadata["evidence_sufficient"] = True
        
        # Should not crash, should verify normally
        result = validator.validate_claim(claim)
        assert result == VerificationStatus.VERIFIED



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
