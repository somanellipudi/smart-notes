"""
Natural Language Inference (NLI) verification for claim-evidence entailment.

This module provides NLI-based verification using RoBERTa-large-MNLI model.

Features:
- Efficient batch processing with configurable batch size
- Proper RoBERTa-MNLI model for accurate classification
- Deterministic output for reproducibility
- Multi-source consensus mode
- Backward compatible single-pair interface

Label mapping (RoBERTa-MNLI):
- 0: Contradiction (evidence contradicts claim)
- 1: Neutral (evidence is irrelevant)
- 2: Entailment (evidence supports claim)
"""

import logging
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import numpy as np

import config

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


class NLIVerifier:
    """
    NLI-based claim verification using RoBERTa-large-MNLI.
    
    Optimized for batch processing and deterministic output.
    
    Usage:
        verifier = NLIVerifier(batch_size=32)
        
        # Single pair (backward compatible)
        result = verifier.verify(claim, evidence)
        
        # Batch pairs (recommended for performance)
        results = verifier.verify_batch([(claim1, evid1), (claim2, evid2)])
        
        # With numpy scores
        results, scores = verifier.verify_batch_with_scores(pairs)
    """
    
    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        device: str = "cpu",
        batch_size: Optional[int] = None
    ):
        """
        Initialize NLI verifier.
        
        Args:
            model_name: Hugging Face model ID
            device: 'cpu' or 'cuda'
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size if batch_size is not None else getattr(config, "NLI_BATCH_SIZE", 1))
        self.use_fp16 = bool(getattr(config, "INFERENCE_FP16", True)) and self.device == "cuda"
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load RoBERTa-MNLI model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info(f"Loading NLI model: {self.model_name} on {self.device}")
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
        Verify single claim-evidence pair.
        
        Backward compatible interface that internally uses batch processing.
        
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
        Processes pairs in batches to balance speed and memory usage.
        
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
            
            # Inference with no gradient computation
            with torch.no_grad():
                if self.use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            
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
        
        Useful for integration with verification pipeline that needs raw scores.
        
        Args:
            pairs: List of (claim, evidence) tuples
        
        Returns:
            Tuple of (results, scores_array)
            - results: List of NLIResult objects
            - scores_array: Shape (len(pairs), 3) with columns [entail, contra, neutral]
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
            Dict with:
            - consensus: bool, True if consensus reached
            - entailment_count: number of entailing sources
            - contradiction_count: number of contradicting sources
            - neutral_count: number of neutral sources
            - results: List of NLIResult objects
            - avg_entailment_prob: average entailment probability
        """
        if not evidence_list:
            return {
                "consensus": False,
                "entailment_count": 0,
                "contradiction_count": 0,
                "neutral_count": 0,
                "results": [],
                "avg_entailment_prob": 0.0
            }
        
        pairs = [(claim, evidence) for evidence in evidence_list]
        results = self.verify_batch(pairs)
        
        entailment_count = sum(
            1 for r in results if r.label == EntailmentLabel.ENTAILMENT
        )
        contradiction_count = sum(
            1 for r in results if r.label == EntailmentLabel.CONTRADICTION
        )
        neutral_count = len(results) - entailment_count - contradiction_count
        
        consensus = (
            entailment_count >= min_entailment_sources
            and contradiction_count == 0
        )
        
        return {
            "consensus": consensus,
            "entailment_count": entailment_count,
            "contradiction_count": contradiction_count,
            "neutral_count": neutral_count,
            "results": results,
            "avg_entailment_prob": np.mean([r.entailment_prob for r in results])
        }
