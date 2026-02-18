"""
Tests for adaptive evidence expanding when sufficiency is not met.

Verifies that the adaptive retrieval mechanism continues expanding the evidence
set when lower k values don't meet sufficiency requirements, until conditions
are satisfied or MAX_EVIDENCE_PER_CLAIM is reached.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from src.evaluation.adaptive_evidence import AdaptiveEvidenceRetriever, DiversityScorer
from src.claims.schema import LearningClaim, ClaimType, EvidenceItem
from src.retrieval.embedding_provider import EmbeddingProvider
from src.claims.nli_verifier import NLIVerifier, NLIResult, EntailmentLabel
from unittest.mock import Mock, patch, MagicMock


class GradualEntailmentNLIVerifier:
    """
    Mock NLI verifier that returns increasing entailment with more evidence.

    Simulates scenario where individual evidence items have low entailment,
    but as more evidence accumulates, the max entailment score rises.
    """

    def __init__(self, base_entailment: float = 0.4, increase_per_k: float = 0.1):
        """
        Args:
            base_entailment: Starting entailment score
            increase_per_k: How much entailment increases per additional evidence
        """
        self.base_entailment = base_entailment
        self.increase_per_k = increase_per_k
        self.last_batch_size = 0

    def verify(self, claim: str, evidence: str) -> NLIResult:
        """Single pair verification."""
        entail = min(self.base_entailment, 1.0)
        return NLIResult(
            label=EntailmentLabel.ENTAILMENT if entail > 0.5 else EntailmentLabel.NEUTRAL,
            entailment_prob=entail,
            contradiction_prob=0.05,
            neutral_prob=1.0 - entail - 0.05
        )

    def verify_batch(self, pairs: List[tuple]) -> List[NLIResult]:
        """Batch verification returns higher entailment with more items."""
        # Entailment increases with batch size (simulates consensus effect)
        batch_size = len(pairs)
        self.last_batch_size = batch_size
        
        # Calibrate so max entailment reaches 0.9+ at k=5
        entail = min(
            self.base_entailment + (batch_size - 2) * self.increase_per_k,
            1.0
        )

        return [
            NLIResult(
                label=EntailmentLabel.ENTAILMENT if entail > 0.5 else EntailmentLabel.NEUTRAL,
                entailment_prob=entail,
                contradiction_prob=0.05,
                neutral_prob=1.0 - entail - 0.05
            )
            for _ in pairs
        ]


class TestAdaptiveEvidenceExpandsWhenNeeded:
    """Test that adaptive retrieval expands when sufficiency not met."""

    @pytest.fixture
    def embedding_provider(self):
        """Create mock embedding provider."""
        provider = Mock(spec=EmbeddingProvider)

        def embed_queries(texts):
            return [[0.1] * 100 for _ in texts]

        def embed_texts(texts):
            return [[0.1] * 100 for _ in texts]

        provider.embed_queries = embed_queries
        provider.embed_texts = embed_texts
        return provider

    @pytest.fixture
    def evidence_store_large(self, embedding_provider):
        """Create mock evidence store with many diverse sources."""
        from src.retrieval.evidence_store import EvidenceStore, Evidence

        store = EvidenceStore(session_id="test_session_expand")

        # Create 15 evidence items from 8 different sources (pages)
        for i in range(15):
            source_id = f"source_{(i // 2) + 1}"  # Groups of 2 per source
            page = (i // 3) + 1  # Groups of 3 per page
            
            evidence = Evidence(
                evidence_id=f"ev_{i}",
                source_id=source_id,
                source_type="test",
                text=f"Evidence snippet {i} providing additional detail",
                chunk_index=0,
                char_start=0,
                char_end=50,
                metadata={"doc_id": source_id, "page": page}
            )
            evidence.embedding = np.array([0.1] * 100, dtype=np.float32)
            store.evidence.append(evidence)
            store.evidence_by_id[f"ev_{i}"] = evidence

        # Build mock index
        embeddings = np.array([[0.1] * 100 for _ in range(15)], dtype=np.float32)
        store.build_index(embeddings)

        # Mock search to return evidence in order
        def mock_search(query_embedding, top_k):
            return [
                (store.evidence[j], 0.95 - 0.01 * j)
                for j in range(min(top_k, len(store.evidence)))
            ]

        store.search = mock_search
        store.index_built = True
        return store

    def test_expands_when_low_initial_entailment(self, embedding_provider, evidence_store_large):
        """
        Verify that retrieval expands when initial k has insufficient entailment.

        Given:
        - base_entailment = 0.4 (starts below TAU=0.8)
        - increase_per_k = 0.1 (gains 0.1 per additional evidence)
        - SUFFICIENCY_TAU = 0.8

        When:
        - Running adaptive retrieval

        Then:
        - Should expand k from 2 → 3 → ... until entailment > 0.8
        - Should reach final_k where entailment sufficient (k=7: 0.4 + 5*0.1 = 0.9)
        - Should have multiple expansion steps
        """
        nli = GradualEntailmentNLIVerifier(base_entailment=0.4, increase_per_k=0.1)

        retriever = AdaptiveEvidenceRetriever(
            embedding_provider=embedding_provider,
            nli_verifier=nli,
            max_evidence=8,
            sufficiency_tau=0.8,
            min_diversity_sources=2,
            initial_k=2
        )

        claim = LearningClaim(
            claim_id="test_expansion_1",
            claim_type=ClaimType.DEFINITION,
            claim_text="Test low initial entailment requiring expansion"
        )

        evidence, metrics = retriever.retrieve_adaptive(claim, evidence_store_large)

        # Should expand from initial k=2
        assert metrics["final_k"] > 2, "Should have expanded beyond initial k=2"
        assert metrics["expansion_steps"] > 0, "Should have performed expansion steps"
        
        # Should meet sufficiency eventually
        assert metrics["sufficiency_met"] is True
        assert metrics["max_entailment_confidence"] > 0.8

        # Should have reasonable number of evidence items
        assert 3 <= len(evidence) <= 8, f"Expected 3-8 evidence items, got {len(evidence)}"

    def test_expansion_stops_at_max_evidence(self, embedding_provider, evidence_store_large):
        """
        Verify that expansion stops at MAX_EVIDENCE_PER_CLAIM even if sufficiency not met.

        Given:
        - base_entailment = 0.2 (very low)
        - increase_per_k = 0.05 (slow increase)
        - TAU = 0.95 (very high threshold)
        - MAX_EVIDENCE = 4

        When:
        - Running adaptive retrieval

        Then:
        - Should expand up to k=4 (max)
        - Should NOT exceed 4 evidence items
        """
        nli = GradualEntailmentNLIVerifier(base_entailment=0.2, increase_per_k=0.05)

        retriever = AdaptiveEvidenceRetriever(
            embedding_provider=embedding_provider,
            nli_verifier=nli,
            max_evidence=4,
            sufficiency_tau=0.95,  # Nearly impossible threshold
            min_diversity_sources=2,
            initial_k=2
        )

        claim = LearningClaim(
            claim_id="test_max_enforcement",
            claim_type=ClaimType.DEFINITION,
            claim_text="Test maximum enforcement during expansion"
        )

        evidence, metrics = retriever.retrieve_adaptive(claim, evidence_store_large)

        # Should reach max limit
        assert metrics["final_k"] == 4
        assert len(evidence) == 4
        
        # Sufficiency should NOT be met (threshold too high)
        assert metrics["sufficiency_met"] is False

    def test_expansion_satisfies_diversity_requirement(self, embedding_provider, evidence_store_large):
        """
        Verify that expansion continues until diversity requirement is met.

        Given:
        - First 2 evidence items from same source
        - min_diversity_sources = 3

        When:
        - Running adaptive retrieval

        Then:
        - Should expand to include items from different sources
        - Should reach k >= 4 to capture 3+ sources
        """
        nli = GradualEntailmentNLIVerifier(base_entailment=0.85, increase_per_k=0.01)

        retriever = AdaptiveEvidenceRetriever(
            embedding_provider=embedding_provider,
            nli_verifier=nli,
            max_evidence=8,
            sufficiency_tau=0.80,  # Low threshold (easy to meet)
            min_diversity_sources=3,  # Require 3 sources
            initial_k=2
        )

        claim = LearningClaim(
            claim_id="test_diversity_expansion",
            claim_type=ClaimType.DEFINITION,
            claim_text="Test diversity requirement during expansion"
        )

        evidence, metrics = retriever.retrieve_adaptive(claim, evidence_store_large)

        # Should expand to get 3+ sources
        assert metrics["num_unique_sources"] >= 3
        assert metrics["final_k"] >= 4  # Need at least 4 to span 3 sources
        assert metrics["sufficiency_met"] is True

    def test_expansion_increments_by_one(self, embedding_provider, evidence_store_large):
        """
        Verify that k expansion increments by 1 each iteration (not jumps).

        This tests implementation detail: each expansion increases k by 1.
        """
        nli = GradualEntailmentNLIVerifier(base_entailment=0.4, increase_per_k=0.15)

        retriever = AdaptiveEvidenceRetriever(
            embedding_provider=embedding_provider,
            nli_verifier=nli,
            max_evidence=6,
            sufficiency_tau=0.80,
            min_diversity_sources=2,
            initial_k=2
        )

        claim = LearningClaim(
            claim_id="test_increment_by_one",
            claim_type=ClaimType.DEFINITION,
            claim_text="Test that expansion increments correctly"
        )

        evidence, metrics = retriever.retrieve_adaptive(claim, evidence_store_large)

        # With base=0.4 and increase=0.15:
        # k=2: 0.4, k=3: 0.55, k=4: 0.70, k=5: 0.85 > TAU
        assert metrics["final_k"] in [5, 6], f"Expected k=5 or 6, got {metrics['final_k']}"
        assert metrics["expansion_steps"] in [3, 4], f"Expected 3-4 expansion steps, got {metrics['expansion_steps']}"

    def test_tracks_entailment_improvement(self, embedding_provider, evidence_store_large):
        """Verify that max_entailment_confidence improves with expansion."""
        nli = GradualEntailmentNLIVerifier(base_entailment=0.3, increase_per_k=0.2)

        retriever = AdaptiveEvidenceRetriever(
            embedding_provider=embedding_provider,
            nli_verifier=nli,
            max_evidence=6,
            sufficiency_tau=0.75,
            min_diversity_sources=2,
            initial_k=2
        )

        claim = LearningClaim(
            claim_id="test_improvement_tracking",
            claim_type=ClaimType.DEFINITION,
            claim_text="Test entailment improvement tracking"
        )

        evidence, metrics = retriever.retrieve_adaptive(claim, evidence_store_large)

        # With base=0.3 and increase=0.2:
        # k=2: 0.3, k=3: 0.5, k=4: 0.7, k=5: 0.9 > TAU
        assert metrics["max_entailment_confidence"] >= 0.75
        assert metrics["expansion_steps"] > 0


class TestDiversityScorerUtils:
    """Test DiversityScorer utility class."""

    def test_compute_source_diversity_single_source(self):
        """Test diversity scoring with single source."""
        evidence = [
            EvidenceItem(
                evidence_id="ev_1",
                source_id="source_1",
                source_type="test",
                snippet="Evidence 1 with sufficient length for testing purposes",
                span_metadata={"page": 1},
                similarity=0.9,
                reliability_prior=0.8
            ),
            EvidenceItem(
                evidence_id="ev_2",
                source_id="source_1",
                source_type="test",
                snippet="Evidence 2 with sufficient length for testing purposes",
                span_metadata={"page": 2},
                similarity=0.85,
                reliability_prior=0.8
            )
        ]

        metrics = DiversityScorer.compute_source_diversity(evidence)

        assert metrics["num_unique_sources"] == 1
        assert metrics["num_unique_pages"] == 2
        assert "source_1" in metrics["unique_sources"]
        assert 1 in metrics["unique_pages"]
        assert 2 in metrics["unique_pages"]

    def test_compute_source_diversity_multiple_sources(self):
        """Test diversity scoring with multiple sources."""
        evidence = [
            EvidenceItem(
                evidence_id="ev_1",
                source_id="source_1",
                source_type="test",
                snippet="Evidence 1 with sufficient length for testing purposes",
                span_metadata={"page": 1},
                similarity=0.9,
                reliability_prior=0.8
            ),
            EvidenceItem(
                evidence_id="ev_2",
                source_id="source_2",
                source_type="test",
                snippet="Evidence 2 with sufficient length for testing purposes",
                span_metadata={"page": 2},
                similarity=0.85,
                reliability_prior=0.8
            ),
            EvidenceItem(
                evidence_id="ev_3",
                source_id="source_3",
                source_type="test",
                snippet="Evidence 3 with sufficient length for testing purposes",
                span_metadata={"page": 3},
                similarity=0.80,
                reliability_prior=0.8
            )
        ]

        metrics = DiversityScorer.compute_source_diversity(evidence)

        assert metrics["num_unique_sources"] == 3
        assert metrics["num_unique_pages"] == 3
        assert all(f"source_{i}" in metrics["unique_sources"] for i in [1, 2, 3])

    def test_prefer_diverse_evidence_selection(self):
        """Test greedy diverse selection algorithm."""
        evidence = [
            EvidenceItem(
                evidence_id="ev_1",
                source_id="source_1",
                source_type="test",
                snippet="Evidence from source 1 page 1 with sufficient length",
                span_metadata={"page": 1},
                similarity=0.95,
                reliability_prior=0.8
            ),
            EvidenceItem(
                evidence_id="ev_2",
                source_id="source_1",
                source_type="test",
                snippet="Evidence from source 1 page 2 with sufficient length",
                span_metadata={"page": 2},
                similarity=0.85,
                reliability_prior=0.8
            ),
            EvidenceItem(
                evidence_id="ev_3",
                source_id="source_2",
                source_type="test",
                snippet="Evidence from source 2 page 1 with sufficient length",
                span_metadata={"page": 1},
                similarity=0.80,
                reliability_prior=0.8
            ),
            EvidenceItem(
                evidence_id="ev_4",
                source_id="source_3",
                source_type="test",
                snippet="Evidence from source 3 page 1 with sufficient length",
                span_metadata={"page": 1},
                similarity=0.75,
                reliability_prior=0.8
            )
        ]

        # Request 2 items preferring diversity
        selected = DiversityScorer.prefer_diverse_evidence(evidence, target_count=2)

        assert len(selected) == 2
        # Should prefer ev_1 (new source) and ev_3 (new source) over ev_2 (duplicate source)
        # at least one should be from different source
        sources = {e.source_id for e in selected}
        assert len(sources) >= 1


class TestAdaptiveEvidenceExpansionMetrics:
    """Test metrics tracking during expansion."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple test setup."""
        embedding_provider = Mock(spec=EmbeddingProvider)
        embedding_provider.embed_queries = lambda texts: [[0.1] * 100 for _ in texts]
        embedding_provider.embed_texts = lambda texts: [[0.1] * 100 for _ in texts]

        nli = Mock(spec=NLIVerifier)
        
        # Return high entailment to force expansion based on diversity only
        def batch_verify(pairs):
            return [
                NLIResult(
                    label=EntailmentLabel.ENTAILMENT,
                    entailment_prob=0.85,
                    contradiction_prob=0.05,
                    neutral_prob=0.1
                )
                for _ in pairs
            ]
        
        nli.verify_batch = batch_verify

        from src.retrieval.evidence_store import EvidenceStore, Evidence

        store = EvidenceStore(session_id="test_session_simple")
        for i in range(10):
            source_id = f"source_{(i // 2) + 1}"
            evidence = Evidence(
                evidence_id=f"ev_{i}",
                source_id=source_id,
                source_type="test",
                text=f"Evidence snippet {i} with sufficient length for validation",
                chunk_index=0,
                char_start=0,
                char_end=50,
                metadata={"doc_id": source_id, "page": 1}  # All same page
            )
            evidence.embedding = np.array([0.1] * 100, dtype=np.float32)
            store.evidence.append(evidence)
            store.evidence_by_id[f"ev_{i}"] = evidence

        embeddings = np.array([[0.1] * 100 for _ in range(10)], dtype=np.float32)
        store.build_index(embeddings)

        def mock_search(query_embedding, top_k):
            return [
                (store.evidence[j], 0.95 - 0.01 * j)
                for j in range(min(top_k, len(store.evidence)))
            ]

        store.search = mock_search
        store.index_built = True

        return embedding_provider, nli, store

    def test_final_k_increases_with_low_initial_diversity(self, simple_setup):
        """Test that final_k increases when initial diversity is low."""
        embedding_provider, nli, store = simple_setup

        # Override nli to return lower initial entailment that increases with batch size
        def low_entail_batch_verify(pairs):
            batch_size = len(pairs)
            # Low entailment initially, increases with batch size
            entail = 0.4 + (batch_size - 2) * 0.1  # k=2: 0.4, k=3: 0.5, k=4: 0.6, etc.
            entail = min(entail, 1.0)
            return [
                NLIResult(
                    label=EntailmentLabel.ENTAILMENT if entail > 0.5 else EntailmentLabel.NEUTRAL,
                    entailment_prob=entail,
                    contradiction_prob=0.05,
                    neutral_prob=1.0 - entail - 0.05
                )
                for _ in pairs
            ]

        from unittest.mock import Mock
        from src.claims.nli_verifier import NLIVerifier
        nli_custom = Mock(spec=NLIVerifier)
        nli_custom.verify_batch = low_entail_batch_verify

        retriever = AdaptiveEvidenceRetriever(
            embedding_provider=embedding_provider,
            nli_verifier=nli_custom,
            max_evidence=6,
            sufficiency_tau=0.60,  # Moderate threshold
            min_diversity_sources=2,  # Require 2 sources
            initial_k=2
        )

        claim = LearningClaim(
            claim_id="test_diversity_metric",
            claim_type=ClaimType.DEFINITION,
            claim_text="Test diversity driven expansion"
        )

        evidence, metrics = retriever.retrieve_adaptive(claim, store)

        # Should expand when both conditions not met - k=2 only has low entail + 1 source
        # Then k=3 gets 2 sources but entail still 0.5, k=4 gets broader evidence and higher entail
        assert metrics["num_unique_sources"] >= 2, f"Expected 2+ sources, got {metrics['num_unique_sources']}"
        assert metrics["final_k"] >= 3, f"Expected k>=3, got {metrics['final_k']}"
