"""
Adaptive Evidence Sufficiency Module

Implements dynamic evidence retrieval that expands/contracts evidence set
based on entailment confidence and diversity metrics rather than fixed top-k.

Features:
- Start with k=2, increase until sufficiency condition met
- Diversity scoring: prefer evidence from different sources/pages
- Entailment confidence thresholding against SUFFICIENCY_TAU
- Maximum evidence cap (MAX_EVIDENCE_PER_CLAIM)

Algorithm:
1. Start with k=2
2. Retrieve top-k evidence
3. Score diversity (source_id/page uniqueness)
4. Compute max entailment confidence from batch NLI
5. Check: if confidence > SUFFICIENCY_TAU AND diversity >= min_sources:
   - Stop and return current evidence
6. Else if k < MAX:
   - Increment k and repeat from step 2
7. Else:
   - Return current evidence (reached max)

See docs/ADAPTIVE_EVIDENCE.md for formal specification.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from src.claims.schema import EvidenceItem, LearningClaim

logger = logging.getLogger(__name__)


class AdaptiveEvidenceRetriever:
    """Retrieve evidence adaptively based on sufficiency metrics."""

    def __init__(
        self,
        embedding_provider: Any,
        nli_verifier: Any,
        max_evidence: int = 6,
        sufficiency_tau: float = 0.8,
        min_diversity_sources: int = 2,
        initial_k: int = 2,
    ):
        """
        Initialize adaptive retriever.

        Args:
            embedding_provider: Provider for embedding queries
            nli_verifier: NLI verifier for entailment checking
            max_evidence: Maximum number of evidence items to retrieve
            sufficiency_tau: Entailment confidence threshold for sufficiency
            min_diversity_sources: Minimum unique sources required
            initial_k: Starting retrieval k value
        """
        self.embedding_provider = embedding_provider
        self.nli_verifier = nli_verifier
        self.max_evidence = max_evidence
        self.sufficiency_tau = sufficiency_tau
        self.min_diversity_sources = min_diversity_sources
        self.initial_k = initial_k

    def retrieve_adaptive(
        self,
        claim: LearningClaim,
        evidence_store: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[EvidenceItem], Dict[str, Any]]:
        """
        Retrieve evidence adaptively until sufficiency condition met.

        Returns:
            Tuple of (evidence_items, metrics_dict)
            where metrics_dict contains:
            - initial_k: Starting k value
            - final_k: Final k retrieved
            - max_entailment_confidence: Highest entailment score found
            - diversity_score: Source uniqueness score
            - num_unique_sources: Number of unique source_ids
            - num_unique_pages: Number of unique pages
            - sufficiency_met: Whether sufficiency condition was satisfied
            - expansion_steps: Number of k expansion iterations
        """
        config = config or {}
        metrics = {
            "initial_k": self.initial_k,
            "final_k": self.initial_k,
            "max_entailment_confidence": 0.0,
            "diversity_score": 0.0,
            "num_unique_sources": 0,
            "num_unique_pages": 0,
            "sufficiency_met": False,
            "expansion_steps": 0,
        }

        current_k = self.initial_k
        best_evidence = []
        best_confidence = 0.0

        while current_k <= self.max_evidence:
            logger.debug(f"Adaptive retrieval: attempting k={current_k}")

            # Retrieve top-k evidence
            evidence = self._retrieve_top_k(claim, evidence_store, current_k)

            if not evidence:
                logger.debug("No evidence retrieved")
                break

            # Score diversity
            diversity_score, num_sources, num_pages = self._score_diversity(
                evidence
            )

            # Compute entailment confidence
            max_confidence = self._compute_max_entailment_confidence(
                claim, evidence, config
            )

            logger.debug(
                f"  k={current_k}: confidence={max_confidence:.3f}, "
                f"diversity={diversity_score:.3f}, "
                f"sources={num_sources}, pages={num_pages}"
            )

            # Update best if better
            if max_confidence > best_confidence:
                best_confidence = max_confidence
                best_evidence = evidence
                metrics["final_k"] = current_k
                metrics["max_entailment_confidence"] = max_confidence
                metrics["diversity_score"] = diversity_score
                metrics["num_unique_sources"] = num_sources
                metrics["num_unique_pages"] = num_pages

            # Check sufficiency condition:
            # confidence > TAU AND (diversity >= min_sources OR reached max)
            diversity_met = (
                num_sources >= self.min_diversity_sources
                or num_pages >= self.min_diversity_sources
            )
            confidence_met = max_confidence > self.sufficiency_tau

            if confidence_met and diversity_met:
                logger.debug(
                    f"Sufficiency condition met at k={current_k}: "
                    f"confidence={max_confidence:.3f} > {self.sufficiency_tau}, "
                    f"diversity={diversity_score:.3f}"
                )
                metrics["sufficiency_met"] = True
                return evidence, metrics

            # Check if reached max
            if current_k >= self.max_evidence:
                logger.debug(
                    f"Reached maximum evidence limit (k={current_k}), "
                    f"returning best found"
                )
                return best_evidence, metrics

            # Expand k by 1
            current_k += 1
            metrics["expansion_steps"] += 1

        logger.debug(
            f"Adaptive retrieval complete: final_k={metrics['final_k']}, "
            f"sufficiency_met={metrics['sufficiency_met']}"
        )
        return best_evidence, metrics

    def _retrieve_top_k(
        self,
        claim: LearningClaim,
        evidence_store: Any,
        k: int
    ) -> List[EvidenceItem]:
        """Retrieve top-k evidence from store."""
        claim_embedding = self.embedding_provider.embed_queries(
            [claim.claim_text]
        )[0]

        if not evidence_store.index_built:
            return []

        search_results = evidence_store.search(claim_embedding, top_k=k)

        evidence_items = []
        for ev, similarity in search_results:
            evidence_item = EvidenceItem(
                evidence_id=ev.evidence_id,
                source_id=ev.source_id,
                source_type=ev.source_type,
                snippet=ev.text[:200],
                span_metadata={"doc_id": ev.metadata.get("doc_id")},
                similarity=float(similarity),
                reliability_prior=0.8,
            )
            evidence_items.append(evidence_item)

        return evidence_items

    def _score_diversity(
        self, evidence: List[EvidenceItem]
    ) -> Tuple[float, int, int]:
        """
        Score diversity of evidence based on unique sources and pages.

        Returns:
            Tuple of (diversity_score, num_unique_sources, num_unique_pages)
            where diversity_score is normalized by total evidence count.
        """
        if not evidence:
            return 0.0, 0, 0

        # Extract unique sources
        unique_sources = set(ev.source_id for ev in evidence)
        num_sources = len(unique_sources)

        # Extract unique pages from span_metadata
        unique_pages = set()
        for ev in evidence:
            page_num = ev.span_metadata.get("page")
            if page_num is not None:
                unique_pages.add(page_num)

        num_pages = len(unique_pages)

        # Diversity score: (unique_sources + unique_pages) normalized by total evidence
        # Maximum diversity = all evidence from different sources and pages
        max_diversity = len(evidence) * 2  # Both source and page could be unique
        actual_diversity = num_sources + num_pages
        diversity_score = min(actual_diversity / max(max_diversity, 1), 1.0)

        return diversity_score, num_sources, num_pages

    def _compute_max_entailment_confidence(
        self,
        claim: LearningClaim,
        evidence: List[EvidenceItem],
        config: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute maximum entailment confidence across evidence set.

        Strategy: Use batch NLI for efficiency, return max entailment score.
        This represents whether ANY evidence strongly supports the claim.
        """
        if not evidence:
            return 0.0

        config = config or {}

        # Use batch NLI if available and multiple evidence items
        if config.get("use_batch_nli", True) and len(evidence) > 1:
            pairs = [(claim.claim_text, ev.snippet) for ev in evidence]
            results = self.nli_verifier.verify_batch(pairs)
            entailment_scores = [r.entailment_prob for r in results]
            max_confidence = float(np.max(entailment_scores))
        else:
            # Single verification for single evidence or when batch disabled
            result = self.nli_verifier.verify(claim.claim_text, evidence[0].snippet)
            max_confidence = float(result.entailment_prob)

        return max_confidence


class DiversityScorer:
    """Utility for computing evidence diversity metrics."""

    @staticmethod
    def compute_source_diversity(
        evidence: List[EvidenceItem],
    ) -> Dict[str, Any]:
        """
        Compute detailed source diversity metrics.

        Returns dict with:
        - unique_sources: Set of unique source_ids
        - unique_pages: Set of unique page numbers
        - source_distribution: Dict of source_id -> count
        - page_distribution: Dict of page -> count
        - entropy: Entropy of source distribution
        """
        source_counts = defaultdict(int)
        page_counts = defaultdict(int)

        for ev in evidence:
            source_counts[ev.source_id] += 1
            page = ev.span_metadata.get("page")
            if page is not None:
                page_counts[page] += 1

        # Compute entropy (measure of distribution uniformity)
        total = len(evidence)
        source_probs = [count / total for count in source_counts.values()]
        entropy = -sum(p * np.log2(p) for p in source_probs if p > 0)

        return {
            "unique_sources": set(source_counts.keys()),
            "unique_pages": set(page_counts.keys()),
            "source_distribution": dict(source_counts),
            "page_distribution": dict(page_counts),
            "num_unique_sources": len(source_counts),
            "num_unique_pages": len(page_counts),
            "entropy": entropy,
        }

    @staticmethod
    def prefer_diverse_evidence(
        evidence: List[EvidenceItem],
        target_count: int
    ) -> List[EvidenceItem]:
        """
        Select target_count items from evidence preferring diversity.

        Strategy: Greedy selection that maximizes source/page coverage.
        Each iteration picks evidence that adds new sources/pages.
        """
        if len(evidence) <= target_count:
            return evidence

        selected = []
        used_sources = set()
        used_pages = set()

        # First pass: always include items from new sources
        for ev in evidence:
            if len(selected) >= target_count:
                break

            page = ev.span_metadata.get("page")
            source_is_new = ev.source_id not in used_sources
            page_is_new = page is not None and page not in used_pages

            if source_is_new or page_is_new:
                selected.append(ev)
                used_sources.add(ev.source_id)
                if page is not None:
                    used_pages.add(page)

        # Second pass: if still need items, add remaining by similarity
        if len(selected) < target_count:
            for ev in evidence:
                if len(selected) >= target_count:
                    break
                if ev not in selected:
                    selected.append(ev)

        return selected
