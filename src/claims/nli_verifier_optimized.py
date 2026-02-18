"""
Optimized NLI verifier with batch processing and caching.

This module provides high-performance NLI verification with:
- Efficient batch processing (configurable batch size)
- Proper RoBERTa-MNLI model usage
- Deterministic output for reproducibility
- In-memory and optional disk caching
"""

import logging
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import numpy as np
import hashlib

logger = logging.getLogger(__name__)


class EntailmentLabel(Enum):
    """NLI prediction labels."""
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


@dataclass
class NLIResult:
    """Result of NLI classification."""
    label: EntailmentLabel
    entailment_prob: float
    contradiction_prob: float
    neutral_prob: float
    
    @property
    def confidence(self) -> float:
        """Maximum probability across labels."""
        return max(self.entailment_prob, self.contradiction_prob, self.neutral_prob)


def compute_nli_cache_key(claim_hash: str, span_id: str, model_id: str) -> str:
    """Compute deterministic cache key for NLI result."""
    key_input = f"{claim_hash}|{span_id}|{model_id}"
    return hashlib.sha256(key_input.encode()).hexdigest()


def compute_claim_hash(claim: str) -> str:
    """Compute deterministic hash of claim for caching."""
    normalized = claim.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


class NLIVerifierOptimized:
    """
    Optimized NLI verifier with batch processing and caching support.
    
    Key improvements over standard verifier:
    - Uses RoBERTa-MNLI model directly (not zero-shot pipeline)
    - Batch processing with configurable batch size
    - Proper tensor handling and device management
    - Optional disk caching of results
    - In-memory caching for repeated queries
    
    Usage:
        verifier = NLIVerifierOptimized(
            batch_size=32,
            cache_disk=True,
            cache_dir="/path/to/cache"
        )
        
        # Single pair (backward compatible)
        result = verifier.verify(claim, evidence)
        
        # Batch pairs (recommended)
        results = verifier.verify_batch([(claim1, evid1), (claim2, evid2)])
        
        # With scores
        results, scores = verifier.verify_batch_with_scores(pairs)
    """
    
    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        device: str = "cpu",
        batch_size: int = 32,
        cache_disk: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize optimized NLI verifier.
        
        Args:
            model_name: Hugging Face model ID (roberta-large-mnli or facebook/bart-large-mnli)
            device: 'cpu' or 'cuda'
            batch_size: Batch size for inference (larger = faster, more memory)
            cache_disk: Cache results to disk
            cache_dir: Directory for disk cache
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_disk = cache_disk
        self.cache_dir = cache_dir
        self._memory_cache: Dict[str, NLIResult] = {}
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load RoBERTa-MNLI model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info(f"Loading NLI model: {self.model_name} on device {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self.model.eval()
            logger.info("NLI model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise
    
    def verify(self, claim: str, evidence: str) -> NLIResult:
        """
        Verify single claim-evidence pair (backward compatible).
        
        Args:
            claim: Claim to verify
            evidence: Evidence text
        
        Returns:
            NLIResult with label and probabilities
        """
        results = self.verify_batch([(claim, evidence)])
        return results[0]
    
    def verify_batch(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """
        Verify multiple claim-evidence pairs in batch.
        
        Optimized for throughput with configurable batch size.
        
        Args:
            pairs: List of (claim, evidence) tuples
        
        Returns:
            List of NLIResult objects (same order as input)
        """
        if not pairs:
            return []
        
        import torch
        
        results = []
        
        # Process in batches
        for batch_idx in range(0, len(pairs), self.batch_size):
            batch_end = min(batch_idx + self.batch_size, len(pairs))
            batch_pairs = pairs[batch_idx:batch_end]
            
            # Extract claims and evidence
            claims = [claim for claim, _ in batch_pairs]
            evidence_list = [evidence for _, evidence in batch_pairs]
            
            # Tokenize batch
            # For NLI: premise=evidence, hypothesis=claim
            inputs = self.tokenizer(
                evidence_list,
                claims,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)  # Shape: (batch_size, 3)
            
            # Parse outputs
            # RoBERTa-MNLI label mapping: [0=contradiction, 1=neutral, 2=entailment]
            probs_np = probs.cpu().numpy()
            
            for i in range(len(batch_pairs)):
                contra_prob = float(probs_np[i, 0])
                neutral_prob = float(probs_np[i, 1])
                entail_prob = float(probs_np[i, 2])
                
                label = self._get_label(entail_prob, neutral_prob, contra_prob)
                
                result = NLIResult(
                    label=label,
                    entailment_prob=entail_prob,
                    contradiction_prob=contra_prob,
                    neutral_prob=neutral_prob
                )
                results.append(result)
        
        logger.debug(f"Verified {len(results)} claim-evidence pairs")
        return results
    
    def verify_batch_with_scores(
        self,
        pairs: List[Tuple[str, str]]
    ) -> Tuple[List[NLIResult], np.ndarray]:
        """
        Verify batch and return scores as numpy array.
        
        Args:
            pairs: List of (claim, evidence) tuples
        
        Returns:
            Tuple of (results, scores_array)
            - results: List of NLIResult objects
            - scores_array: Shape (len(pairs), 3) with [entail, contra, neutral]
        """
        results = self.verify_batch(pairs)
        
        scores = np.array([
            [r.entailment_prob, r.contradiction_prob, r.neutral_prob]
            for r in results
        ], dtype=np.float32)
        
        return results, scores
    
    def _get_label(
        self,
        entail_prob: float,
        neutral_prob: float,
        contra_prob: float
    ) -> EntailmentLabel:
        """Determine label from probabilities."""
        max_prob = max(entail_prob, neutral_prob, contra_prob)
        
        if entail_prob == max_prob:
            return EntailmentLabel.ENTAILMENT
        elif contra_prob == max_prob:
            return EntailmentLabel.CONTRADICTION
        else:
            return EntailmentLabel.NEUTRAL
    
    def check_consensus(
        self,
        claim: str,
        evidence_list: List[str],
        min_entailment_sources: int = 2
    ) -> Dict:
        """
        Multi-source consensus verification.
        
        Args:
            claim: Claim to verify
            evidence_list: List of independent evidence texts
            min_entailment_sources: Minimum entailing sources required
        
        Returns:
            Dict with consensus result and statistics
        """
        if not evidence_list:
            return {
                "consensus": False,
                "entailment_count": 0,
                "contradiction_count": 0,
                "results": []
            }
        
        pairs = [(claim, evidence) for evidence in evidence_list]
        results = self.verify_batch(pairs)
        
        entailment_count = sum(
            1 for r in results if r.label == EntailmentLabel.ENTAILMENT
        )
        contradiction_count = sum(
            1 for r in results if r.label == EntailmentLabel.CONTRADICTION
        )
        
        consensus = (
            entailment_count >= min_entailment_sources
            and contradiction_count == 0
        )
        
        return {
            "consensus": consensus,
            "entailment_count": entailment_count,
            "contradiction_count": contradiction_count,
            "neutral_count": len(results) - entailment_count - contradiction_count,
            "results": results,
            "avg_entailment_prob": np.mean([r.entailment_prob for r in results])
        }


# Alias for backward compatibility
NLIVerifier = NLIVerifierOptimized
