"""
Tests for adaptive evidence stopping early when sufficiency condition is met.

Verifies that the adaptive retrieval mechanism stops expanding the evidence set
when sufficiency thresholds are reached, without retrieving all MAX_EVIDENCE_PER_CLAIM.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, List, Any

from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner
from src.evaluation.adaptive_evidence import AdaptiveEvidenceRetriever
from src.claims.schema import LearningClaim, ClaimType, EvidenceItem
from src.retrieval.embedding_provider import EmbeddingProvider
from src.claims.nli_verifier import NLIVerifier, NLIResult, EntailmentLabel
from unittest.mock import Mock, patch, MagicMock


class MockNLIVerifier:
    """Mock NLI verifier that returns high entailment on first call."""

    def __init__(self, high_entailment_until_k: int = 2):
        """
        Args:
            high_entailment_until_k: Return high entailment until this k value,
                                    then return lower scores
        """
        self.high_entailment_until_k = high_entailment_until_k
        self.calls = []

    def verify(self, claim: str, evidence: str) -> NLIResult:
        """Single pair verification."""
        self.calls.append(("single", claim, evidence))
        # Return high entailment for early stops
        return NLIResult(
            label=EntailmentLabel.ENTAILMENT,
            entailment_prob=0.9,
            contradiction_prob=0.05,
            neutral_prob=0.05
        )

    def verify_batch(self, pairs: List[tuple]) -> List[NLIResult]:
        """Batch verification returns high entailment."""
        self.calls.append(("batch", len(pairs)))
        return [
            NLIResult(
                label=EntailmentLabel.ENTAILMENT,
                entailment_prob=0.9,
                contradiction_prob=0.05,
                neutral_prob=0.05
            )
            for _ in pairs
        ]


class TestAdaptiveEvidenceStopsEarly:
    """Test that adaptive retrieval stops when sufficiency met."""

    @pytest.fixture
    def embedding_provider(self):
        """Create mock embedding provider."""
        provider = Mock(spec=EmbeddingProvider)

        def embed_queries(texts):
            # Return dummy embeddings
            return [[0.1] * 100 for _ in texts]

        def embed_texts(texts):
            return [[0.1] * 100 for _ in texts]

        provider.embed_queries = embed_queries
        provider.embed_texts = embed_texts
        return provider

    @pytest.fixture
    def nli_verifier(self):
        """Create mock NLI verifier with high entailment."""
        return MockNLIVerifier(high_entailment_until_k=2)

    @pytest.fixture
    def evidence_store(self, embedding_provider):
        """Create mock evidence store with diverse sources."""
        from src.retrieval.evidence_store import EvidenceStore, Evidence
        import numpy as np

        store = EvidenceStore(session_id="test_session_stops_early")

        # Create 10 evidence items - arrange so k=2 gets 2 different sources
        # ev_0 -> source_1, ev_1 -> source_2, ev_2 -> source_3, etc.
        for i in range(10):
            source_id = f"source_{i + 1}"  # Each item from different source
            evidence = Evidence(
                evidence_id=f"ev_{i}",
                source_id=source_id,
                source_type="test",
                text=f"Evidence snippet {i} supporting the claim",
                chunk_index=0,
                char_start=0,
                char_end=50,
                metadata={"doc_id": source_id, "page": 1}
            )
            evidence.embedding = np.array([0.1] * 100, dtype=np.float32)
            store.evidence.append(evidence)
            store.evidence_by_id[f"ev_{i}"] = evidence

        # Build mock index
        embeddings = np.array([[0.1] * 100 for _ in range(10)], dtype=np.float32)
        store.build_index(embeddings)

        # Mock search to return evidence in order
        def mock_search(query_embedding, top_k):
            return [(store.evidence[i], 0.9 - 0.01 * i) for i in range(min(top_k, len(store.evidence)))]

        store.search = mock_search
        store.index_built = True
        return store

    def test_stops_early_with_sufficient_confidence(self, embedding_provider, nli_verifier, evidence_store):
        """
        Verify that adaptive retrieval stops at k=2 when sufficiency condition is met.

        Given:
        - MAX_EVIDENCE_PER_CLAIM = 6
        - SUFFICIENCY_TAU = 0.8
        - EVIDENCE_DIVERSITY_MIN_SOURCES = 2
        - High entailment (0.9) returned by NLI

        When:
        - Running adaptive retrieval with enforced high entailment

        Then:
        - Should stop at k=2 (sufficiency met)
        - Should NOT retrieve all 6 evidence items
        """
        retriever = AdaptiveEvidenceRetriever(
            embedding_provider=embedding_provider,
            nli_verifier=nli_verifier,
            max_evidence=6,
            sufficiency_tau=0.8,
            min_diversity_sources=2,
            initial_k=2
        )

        claim = LearningClaim(
            claim_id="test_claim_1",
            claim_type=ClaimType.DEFINITION,
            claim_text="Python is a high-level programming language"
        )

        evidence, metrics = retriever.retrieve_adaptive(claim, evidence_store)

        # Verify stopped early
        assert metrics["sufficiency_met"] is True
        assert metrics["final_k"] == 2, f"Expected final_k=2, got {metrics['final_k']}"
        assert len(evidence) == 2, f"Expected 2 evidence items, got {len(evidence)}"
        assert metrics["max_entailment_confidence"] > 0.8
        assert metrics["num_unique_sources"] >= 2

    def test_respects_sufficiency_tau_threshold(self, embedding_provider, nli_verifier, evidence_store):
        """
        Verify that sufficiency tau threshold is respected:
        - If max entailment < TAU, continue expanding
        - If max entailment > TAU, can stop (if diversity also met)
        """
        retriever = AdaptiveEvidenceRetriever(
            embedding_provider=embedding_provider,
            nli_verifier=nli_verifier,
            max_evidence=6,
            sufficiency_tau=0.85,  # High threshold (but still less than 0.9 returned)
            min_diversity_sources=2,
            initial_k=2
        )

        claim = LearningClaim(
            claim_id="test_claim_2",
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim for tau threshold"
        )

        evidence, metrics = retriever.retrieve_adaptive(claim, evidence_store)

        # With 0.9 entailment and TAU=0.85, should still stop early
        assert metrics["sufficiency_met"] is True
        assert metrics["max_entailment_confidence"] > 0.85

    def test_expansion_steps_tracked(self, embedding_provider, nli_verifier, evidence_store):
        """Verify that expansion steps are tracked correctly."""
        retriever = AdaptiveEvidenceRetriever(
            embedding_provider=embedding_provider,
            nli_verifier=nli_verifier,
            max_evidence=6,
            sufficiency_tau=0.8,
            min_diversity_sources=2,
            initial_k=2
        )

        claim = LearningClaim(
            claim_id="test_claim_3",
            claim_type=ClaimType.DEFINITION,
            claim_text="Test expansion tracking"
        )

        evidence, metrics = retriever.retrieve_adaptive(claim, evidence_store)

        # With sufficiency met at k=2 (initial_k), expansion_steps should be 0
        assert metrics["expansion_steps"] == 0, f"Expected 0 expansions, got {metrics['expansion_steps']}"
        assert metrics["initial_k"] == 2
        assert metrics["final_k"] == 2

    def test_diversity_requirement_in_early_stop(self, embedding_provider, nli_verifier, evidence_store):
        """
        Verify that diversity is checked when stopping early.

        Given:
        - Two evidence items from different sources (e.g., source_1 and source_2)

        Then:
        - With min_diversity_sources=2, should meet diversity at k=2
        - Should be able to stop early
        """
        retriever = AdaptiveEvidenceRetriever(
            embedding_provider=embedding_provider,
            nli_verifier=nli_verifier,
            max_evidence=6,
            sufficiency_tau=0.8,
            min_diversity_sources=2,  # Require 2 sources
            initial_k=2
        )

        claim = LearningClaim(
            claim_id="test_claim_4",
            claim_type=ClaimType.DEFINITION,
            claim_text="Test diversity requirement"
        )

        evidence, metrics = retriever.retrieve_adaptive(claim, evidence_store)

        # Should meet diversity at k=2 (items from source_1 and source_2)
        assert metrics["num_unique_sources"] >= 2
        assert metrics["sufficiency_met"] is True

    def test_no_unnecessary_retrieval_beyond_max(self, embedding_provider, nli_verifier, evidence_store):
        """
        Verify that retrieval never exceeds MAX_EVIDENCE_PER_CLAIM.

        Even if sufficiency not met, should stop at max.
        """
        # Create low-confidence NLI verifier (will need to expand)
        low_conf_nli = Mock(spec=NLIVerifier)

        def low_conf_verify_batch(pairs):
            return [
                NLIResult(
                    label=EntailmentLabel.NEUTRAL,
                    entailment_prob=0.3,  # Low entailment
                    contradiction_prob=0.1,
                    neutral_prob=0.6
                )
                for _ in pairs
            ]

        low_conf_nli.verify_batch = low_conf_verify_batch

        retriever = AdaptiveEvidenceRetriever(
            embedding_provider=embedding_provider,
            nli_verifier=low_conf_nli,
            max_evidence=4,  # Small max
            sufficiency_tau=0.95,  # Unattainable threshold
            min_diversity_sources=2,
            initial_k=2
        )

        claim = LearningClaim(
            claim_id="test_claim_5",
            claim_type=ClaimType.DEFINITION,
            claim_text="Test max enforcement"
        )

        evidence, metrics = retriever.retrieve_adaptive(claim, evidence_store)

        # Should never retrieve more than max_evidence
        assert len(evidence) <= 4
        assert metrics["final_k"] <= 4

    def test_metrics_completeness(self, embedding_provider, nli_verifier, evidence_store):
        """Verify that all expected metrics are returned."""
        retriever = AdaptiveEvidenceRetriever(
            embedding_provider=embedding_provider,
            nli_verifier=nli_verifier,
            max_evidence=6,
            sufficiency_tau=0.8,
            min_diversity_sources=2,
            initial_k=2
        )

        claim = LearningClaim(
            claim_id="test_claim_6",
            claim_type=ClaimType.DEFINITION,
            claim_text="Test metrics completeness"
        )

        evidence, metrics = retriever.retrieve_adaptive(claim, evidence_store)

        # Verify all expected keys present
        expected_keys = {
            "initial_k",
            "final_k",
            "max_entailment_confidence",
            "diversity_score",
            "num_unique_sources",
            "num_unique_pages",
            "sufficiency_met",
            "expansion_steps",
        }
        assert set(metrics.keys()) == expected_keys

        # Verify value types and ranges
        assert isinstance(metrics["initial_k"], int)
        assert isinstance(metrics["final_k"], int)
        assert 0.0 <= metrics["max_entailment_confidence"] <= 1.0
        assert 0.0 <= metrics["diversity_score"] <= 1.0
        assert metrics["num_unique_sources"] >= 0
        assert metrics["num_unique_pages"] >= 0
        assert isinstance(metrics["sufficiency_met"], bool)
        assert metrics["expansion_steps"] >= 0


class TestAdaptiveEvidenceStopsEarlyIntegration:
    """Integration tests with CSBenchmarkRunner."""

    @pytest.fixture
    def benchmark_dataset_small(self, tmp_path):
        """Create small benchmark dataset for testing."""
        dataset = [
            {
                "doc_id": "test_1",
                "domain_topic": "algorithms",
                "claim": "Quicksort has average time complexity O(n log n)",
                "source_text": "Quicksort average case. Time complexity is O(n log n). "
                              "This is achieved through random pivoting.",
                "evidence_spans": [(0, 100)],
                "gold_label": "ENTAIL"
            },
            {
                "doc_id": "test_2",
                "domain_topic": "data-structures",
                "claim": "A binary search tree must be balanced",
                "source_text": "BSTs don't require balancing. AVL trees are self-balancing variants of BSTs.",
                "evidence_spans": [(0, 50)],
                "gold_label": "CONTRADICT"
            }
        ]

        dataset_file = tmp_path / "test_dataset.jsonl"
        with open(dataset_file, "w") as f:
            for example in dataset:
                f.write(json.dumps(example) + "\n")

        return str(dataset_file)

    def test_integration_adaptive_stops_early(self, benchmark_dataset_small):
        """Test adaptive retrieval integration with benchmark runner."""
        runner = CSBenchmarkRunner(benchmark_dataset_small)

        # Run with adaptive evidence enabled
        config = {
            "use_retrieval": True,
            "use_adaptive_evidence": True,
            "use_batch_nli": True,
            "use_nli": True,
        }

        # Run on single example
        results = runner.run(sample_size=1, config=config)

        # Verify metrics tracked
        assert len(runner.adaptive_metrics_per_claim) > 0

        # Check metrics
        for claim_id, metrics in runner.adaptive_metrics_per_claim.items():
            assert metrics["initial_k"] == 2
            assert metrics["final_k"] <= 6
            assert "sufficiency_met" in metrics
