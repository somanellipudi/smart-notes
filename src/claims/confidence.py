"""
Research-grade confidence scoring for verified claims.

Combines multiple signals:
- Semantic similarity (max)
- Number of independent sources
- Entailment probability (NLI)
- Contradiction penalty
- Graph support depth

Formula:
    confidence = w1*sem_sim + w2*source_diversity + w3*entailment_prob 
                 - w4*contradiction_penalty + w5*graph_support

Calibration-aware with temperature scaling.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceFactors:
    """Individual factors contributing to confidence."""
    max_similarity: float = 0.0
    source_count: int = 0
    source_diversity: float = 0.0  # proportion of unique source types
    entailment_prob: float = 0.0
    contradiction_penalty: float = 0.0
    graph_support: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "max_similarity": self.max_similarity,
            "source_count": self.source_count,
            "source_diversity": self.source_diversity,
            "entailment_prob": self.entailment_prob,
            "contradiction_penalty": self.contradiction_penalty,
            "graph_support": self.graph_support
        }


class ConfidenceCalculator:
    """
    Calculate calibrated confidence scores for claims.
    
    Weights are learned from calibration data or set based on domain.
    Default weights prioritize entailment > similarity > diversity.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0
    ):
        """
        Initialize confidence calculator.
        
        Args:
            weights: Custom weights for factors
            temperature: Temperature scaling for calibration
        """
        self.weights = weights or {
            "similarity": 0.25,
            "source_count": 0.15,
            "source_diversity": 0.10,
            "entailment": 0.35,
            "contradiction": 0.10,
            "graph_support": 0.05
        }
        self.temperature = temperature
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
        logger.info(f"Confidence weights: {self.weights}")
    
    def calculate(
        self,
        factors: ConfidenceFactors,
        apply_temperature: bool = True
    ) -> float:
        """
        Calculate confidence score from factors.
        
        Args:
            factors: Individual confidence factors
            apply_temperature: Whether to apply temperature scaling
        
        Returns:
            Confidence score in [0, 1]
        """
        # Weighted combination
        score = (
            self.weights["similarity"] * factors.max_similarity
            + self.weights["source_count"] * self._normalize_source_count(factors.source_count)
            + self.weights["source_diversity"] * factors.source_diversity
            + self.weights["entailment"] * factors.entailment_prob
            - self.weights["contradiction"] * factors.contradiction_penalty
            + self.weights["graph_support"] * factors.graph_support
        )
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        
        # Apply temperature scaling
        if apply_temperature and self.temperature != 1.0:
            score = self._apply_temperature(score)
        
        return score
    
    def calculate_from_evidence(
        self,
        evidence_list: List[Dict],
        nli_results: Optional[List] = None,
        graph_support: float = 0.0
    ) -> Tuple[float, ConfidenceFactors]:
        """
        Calculate confidence directly from evidence and NLI results.
        
        Args:
            evidence_list: List of evidence dicts with 'similarity', 'source_type'
            nli_results: Optional list of NLI results
            graph_support: Graph-based support score
        
        Returns:
            (confidence_score, factors)
        """
        if not evidence_list:
            return 0.0, ConfidenceFactors()
        
        # Extract factors
        similarities = [e.get("similarity", 0.0) for e in evidence_list]
        max_similarity = max(similarities) if similarities else 0.0
        
        source_types = [e.get("source_type", "") for e in evidence_list]
        source_diversity = len(set(source_types)) / len(source_types) if source_types else 0.0
        
        # NLI factors
        entailment_prob = 0.0
        contradiction_penalty = 0.0
        
        if nli_results:
            entailment_probs = [r.entailment_prob for r in nli_results]
            contradiction_probs = [r.contradiction_prob for r in nli_results]
            
            entailment_prob = max(entailment_probs) if entailment_probs else 0.0
            contradiction_penalty = max(contradiction_probs) if contradiction_probs else 0.0
        
        factors = ConfidenceFactors(
            max_similarity=max_similarity,
            source_count=len(evidence_list),
            source_diversity=source_diversity,
            entailment_prob=entailment_prob,
            contradiction_penalty=contradiction_penalty,
            graph_support=graph_support
        )
        
        confidence = self.calculate(factors)
        
        return confidence, factors
    
    def _normalize_source_count(self, count: int, max_count: int = 10) -> float:
        """Normalize source count to [0, 1]."""
        return min(count / max_count, 1.0)
    
    def _apply_temperature(self, score: float) -> float:
        """Apply temperature scaling for calibration."""
        # Transform score to logit
        epsilon = 1e-7
        score = max(epsilon, min(1 - epsilon, score))
        logit = np.log(score / (1 - score))
        
        # Scale by temperature
        calibrated_logit = logit / self.temperature
        
        # Transform back to probability
        calibrated_score = 1 / (1 + np.exp(-calibrated_logit))
        
        return float(calibrated_score)
    
    def fit_temperature(
        self,
        predictions: List[float],
        labels: List[int],
        validation_split: float = 0.2
    ) -> float:
        """
        Learn optimal temperature from labeled data.
        
        Args:
            predictions: Model confidence scores
            labels: Binary labels (1=correct, 0=incorrect)
            validation_split: Fraction for validation
        
        Returns:
            Optimal temperature
        """
        from scipy.optimize import minimize
        
        # Split data
        n = len(predictions)
        split_idx = int(n * (1 - validation_split))
        
        train_preds = predictions[:split_idx]
        train_labels = labels[:split_idx]
        val_preds = predictions[split_idx:]
        val_labels = labels[split_idx:]
        
        def nll_loss(temp):
            """Negative log-likelihood loss."""
            temp = max(temp, 0.01)  # Prevent division by zero
            scaled_preds = []
            for p in train_preds:
                calibrated = self._apply_temperature_with_temp(p, temp)
                scaled_preds.append(calibrated)
            
            # Binary cross-entropy
            epsilon = 1e-7
            loss = 0
            for pred, label in zip(scaled_preds, train_labels):
                pred = max(epsilon, min(1 - epsilon, pred))
                loss -= label * np.log(pred) + (1 - label) * np.log(1 - pred)
            
            return loss / len(train_labels)
        
        # Optimize temperature
        result = minimize(nll_loss, x0=1.0, method='Nelder-Mead')
        optimal_temp = max(result.x[0], 0.01)
        
        logger.info(f"Optimal temperature: {optimal_temp:.3f}")
        
        # Update instance temperature
        self.temperature = optimal_temp
        
        return optimal_temp
    
    def _apply_temperature_with_temp(self, score: float, temp: float) -> float:
        """Apply specific temperature value."""
        epsilon = 1e-7
        score = max(epsilon, min(1 - epsilon, score))
        logit = np.log(score / (1 - score))
        calibrated_logit = logit / temp
        return float(1 / (1 + np.exp(-calibrated_logit)))


class ConsensusVerifier:
    """
    Multi-source consensus verification for high-stakes domains.
    
    Requires claims to be supported by multiple independent sources
    with entailment agreement.
    """
    
    def __init__(
        self,
        min_sources: int = 2,
        min_entailment_prob: float = 0.7,
        require_diverse_sources: bool = True
    ):
        """
        Initialize consensus verifier.
        
        Args:
            min_sources: Minimum number of independent sources required
            min_entailment_prob: Minimum entailment probability per source
            require_diverse_sources: Whether sources must be from different types
        """
        self.min_sources = min_sources
        self.min_entailment_prob = min_entailment_prob
        self.require_diverse_sources = require_diverse_sources
    
    def verify_consensus(
        self,
        evidence_list: List[Dict],
        nli_results: List
    ) -> Dict[str, any]:
        """
        Check if claim meets consensus requirements.
        
        Args:
            evidence_list: List of evidence items with source metadata
            nli_results: Corresponding NLI results
        
        Returns:
            Dict with consensus status and details
        """
        if len(evidence_list) < self.min_sources:
            return {
                "consensus": False,
                "reason": f"Insufficient sources ({len(evidence_list)} < {self.min_sources})",
                "entailment_count": 0
            }
        
        # Count entailing sources
        entailing_sources = [
            (ev, nli)
            for ev, nli in zip(evidence_list, nli_results)
            if nli.entailment_prob >= self.min_entailment_prob
        ]
        
        if len(entailing_sources) < self.min_sources:
            return {
                "consensus": False,
                "reason": f"Insufficient entailing sources ({len(entailing_sources)} < {self.min_sources})",
                "entailment_count": len(entailing_sources)
            }
        
        # Check source diversity if required
        if self.require_diverse_sources:
            source_types = set(ev.get("source_type") for ev, _ in entailing_sources)
            if len(source_types) < 2:
                return {
                    "consensus": False,
                    "reason": "Sources not diverse (all from same type)",
                    "entailment_count": len(entailing_sources),
                    "source_types": list(source_types)
                }
        
        return {
            "consensus": True,
            "entailment_count": len(entailing_sources),
            "avg_entailment_prob": np.mean([nli.entailment_prob for _, nli in entailing_sources]),
            "source_types": list(set(ev.get("source_type") for ev, _ in entailing_sources))
        }
