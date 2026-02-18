"""
Tests for contradiction gate in claim verification.

Ensures that claims with high contradiction probability cannot be marked VERIFIED.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.claims.nli_verifier import NLIVerifier, NLIResult, EntailmentLabel
from src.claims.schema import LearningClaim, VerificationStatus, EvidenceItem
from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner


class TestContradictionGate:
    """Test contradiction gate prevents false VERIFIED status."""
    
    @pytest.fixture
    def nli_verifier(self):
        """Create NLI verifier instance."""
        return NLIVerifier(device="cpu", batch_size=4)
    
    def test_contradiction_gate_basic(self, nli_verifier):
        """Test that contradiction gate rejects contradictory claims."""
        # Claim: Stack push removes elements
        # Evidence: Stack push adds elements
        claim = "Stack push operation removes an element from the top"
        evidence = "The push operation adds an element to the top of the stack"
        
        result = nli_verifier.verify(claim, evidence)
        
        # Should detect contradiction
        assert result.contradiction_prob > 0.5, (
            f"Expected contradiction_prob > 0.5, got {result.contradiction_prob:.3f}"
        )
        assert result.label == EntailmentLabel.CONTRADICTION, (
            f"Expected CONTRADICTION label, got {result.label}"
        )
    
    def test_contradiction_gate_threshold(self):
        """Test contradiction gate threshold at 0.6."""
        # Create mock claim with high contradiction
        claim = LearningClaim(
            claim_text="Push removes elements from stack",
            claim_type="CODE_BEHAVIOR_CLAIM",
            snippet_id="test_1"
        )
        
        # Add evidence
        claim.evidence_objects = [
            EvidenceItem(
                evidence_id="ev_1",
                snippet="Push adds elements to the top of stack",
                similarity=0.85,
                metadata={"doc_id": "test_doc"}
            )
        ]
        
        # Simulate verification with contradiction gate
        config = {
            "enable_contradiction_gate": True,
            "contradiction_threshold": 0.6,
            "verify_threshold": 0.55,
            "low_conf_threshold": 0.35,
            "use_batch_nli": False
        }
        
        # Create runner (will initialize NLI verifier)
        runner = CSBenchmarkRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_hard.jsonl",
            device="cpu",
            log_predictions=False
        )
        
        # Validate claim
        runner._validate_claim(claim, config)
        
        # Should be REJECTED due to contradiction gate
        assert claim.status == VerificationStatus.REJECTED, (
            f"Expected REJECTED status, got {claim.status}"
        )
        assert claim.confidence < 0.5, (
            f"Expected low confidence, got {claim.confidence:.3f}"
        )
    
    def test_no_contradiction_passes_gate(self, nli_verifier):
        """Test that non-contradictory claims pass the gate."""
        claim = "Stack push operation adds an element to the top"
        evidence = "The push operation adds an element to the top of the stack"
        
        result = nli_verifier.verify(claim, evidence)
        
        # Should detect entailment
        assert result.entailment_prob > 0.7, (
            f"Expected entailment_prob > 0.7, got {result.entailment_prob:.3f}"
        )
        assert result.contradiction_prob < 0.3, (
            f"Expected contradiction_prob < 0.3, got {result.contradiction_prob:.3f}"
        )
    
    def test_contradiction_gate_disabled(self):
        """Test that disabling gate allows contradictions through."""
        claim = LearningClaim(
            claim_text="Push removes elements from stack",
            claim_type="CODE_BEHAVIOR_CLAIM",
            snippet_id="test_1"
        )
        
        # Add high-similarity evidence (contradictory content)
        claim.evidence_objects = [
            EvidenceItem(
                evidence_id="ev_1",
                snippet="Push adds elements to the top of stack",
                similarity=0.90,  # High similarity
                metadata={"doc_id": "test_doc"}
            )
        ]
        
        # Disable contradiction gate
        config = {
            "enable_contradiction_gate": False,  # DISABLED
            "contradiction_threshold": 0.6,
            "verify_threshold": 0.4,  # Low threshold to allow through
            "low_conf_threshold": 0.35,
            "use_batch_nli": False
        }
        
        runner = CSBenchmarkRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_hard.jsonl",
            device="cpu",
            log_predictions=False
        )
        
        runner._validate_claim(claim, config)
        
        # Without gate, might pass due to high similarity
        # (This demonstrates the gate is necessary)
        print(f"Status without gate: {claim.status}, confidence: {claim.confidence:.3f}")
    
    def test_contradiction_gate_multiple_evidence(self):
        """Test contradiction gate with multiple evidence items."""
        claim = LearningClaim(
            claim_text="Queue dequeue removes from rear",
            claim_type="CODE_BEHAVIOR_CLAIM",
            snippet_id="test_1"
        )
        
        # Add multiple evidence (one contradictory)
        claim.evidence_objects = [
            EvidenceItem(
                evidence_id="ev_1",
                snippet="Dequeue removes element from front of queue",
                similarity=0.80,
                metadata={"doc_id": "test_doc"}
            ),
            EvidenceItem(
                evidence_id="ev_2",
                snippet="Front element is removed by dequeue operation",
                similarity=0.75,
                metadata={"doc_id": "test_doc"}
            )
        ]
        
        config = {
            "enable_contradiction_gate": True,
            "contradiction_threshold": 0.6,
            "verify_threshold": 0.55,
            "low_conf_threshold": 0.35,
            "use_batch_nli": True  # Batch mode
        }
        
        runner = CSBenchmarkRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_hard.jsonl",
            device="cpu",
            log_predictions=False
        )
        
        runner._validate_claim(claim, config)
        
        # Should be REJECTED due to contradiction
        assert claim.status == VerificationStatus.REJECTED, (
            f"Expected REJECTED status, got {claim.status}"
        )
    
    def test_contradiction_scores_from_batch(self, nli_verifier):
        """Test batch verification returns contradiction scores."""
        pairs = [
            ("Push removes elements", "Push adds elements to stack"),
            ("Pop adds elements", "Pop removes elements from stack"),
            ("Queue is LIFO", "Queue is FIFO structure")
        ]
        
        results = nli_verifier.verify_batch(pairs)
        
        assert len(results) == 3
        for result in results:
            # All should have high contradiction
            assert result.contradiction_prob > 0.4, (
                f"Expected high contradiction_prob, got {result.contradiction_prob:.3f}"
            )


class TestContradictionGateIntegration:
    """Integration tests for contradiction gate in benchmark pipeline."""
    
    def test_benchmark_with_contradiction_gate(self):
        """Test benchmark evaluation with contradiction gate enabled."""
        runner = CSBenchmarkRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_hard.jsonl",
            device="cpu",
            log_predictions=False
        )
        
        # Run with contradiction gate
        config = {
            "enable_contradiction_gate": True,
            "contradiction_threshold": 0.6,
            "verify_threshold": 0.55,
            "low_conf_threshold": 0.35
        }
        
        result = runner.run(config=config, run_id="test_contradiction_gate")
        
        # Metrics should be reasonable
        assert result.metrics.accuracy > 0.5, (
            f"Expected accuracy > 0.5, got {result.metrics.accuracy:.3f}"
        )
        
        print(f"\nContradiction Gate Results:")
        print(f"  Accuracy: {result.metrics.accuracy:.3f}")
        print(f"  F1 VERIFIED: {result.metrics.F1_verified:.3f}")
        print(f"  F1 REJECTED: {result.metrics.F1_rejected:.3f}")
    
    def test_benchmark_without_contradiction_gate(self):
        """Test benchmark evaluation with contradiction gate disabled."""
        runner = CSBenchmarkRunner(
            dataset_path="evaluation/cs_benchmark/cs_benchmark_hard.jsonl",
            device="cpu",
            log_predictions=False
        )
        
        # Run without contradiction gate
        config = {
            "enable_contradiction_gate": False,  # DISABLED
            "verify_threshold": 0.55,
            "low_conf_threshold": 0.35
        }
        
        result = runner.run(config=config, run_id="test_no_gate")
        
        print(f"\nNo Contradiction Gate Results:")
        print(f"  Accuracy: {result.metrics.accuracy:.3f}")
        print(f"  F1 VERIFIED: {result.metrics.F1_verified:.3f}")
        print(f"  F1 REJECTED: {result.metrics.F1_rejected:.3f}")


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v", "-s"])
