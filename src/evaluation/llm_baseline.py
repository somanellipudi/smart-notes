"""
LLM Baseline Wrapper for Fact-Checking.

Provides a retrieval + LLM judge baseline for comparison:
- Retrieves evidence using the same retrieval system
- Prompts LLM to judge claim veracity
- Returns label + confidence score

Supports:
- GPT-4o (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- Llama 3 (local or API)
- Deterministic stub (when no API keys available)

If no API key exists, falls back to deterministic stub for testing.
"""

import json
import logging
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict

from sklearn.metrics import f1_score

from src.evaluation.calibration import CalibrationEvaluator
from src.evaluation.selective_prediction import compute_risk_coverage_curve, compute_auc_accuracy_coverage

logger = logging.getLogger(__name__)


@dataclass
class LLMPrediction:
    """Single LLM prediction."""
    claim_id: str
    claim_text: str
    predicted_label: str
    confidence: float
    reasoning: str
    evidence_used: List[str]
    latency_ms: float
    retrieval_ms: float = 0.0
    inference_ms: float = 0.0


@dataclass
class LLMBaselineResult:
    """Results from LLM baseline evaluation."""
    model_name: str
    accuracy: float
    macro_f1: float
    ece: float
    auc_ac: float
    avg_latency_ms: float
    total_cost_usd: float
    stub_mode: bool
    baseline_note: str
    predictions: List[LLMPrediction]
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "ece": self.ece,
            "auc_ac": self.auc_ac,
            "avg_latency_ms": self.avg_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "stub_mode": self.stub_mode,
            "baseline_note": self.baseline_note,
            "predictions": [asdict(p) for p in self.predictions]
        }


class LLMBaseline:
    """
    LLM-based baseline for claim verification.
    
    Uses retrieval + LLM prompting to judge claims.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        use_stub: bool = False
    ):
        """
        Initialize LLM baseline.
        
        Args:
            model_name: LLM model to use (gpt-4o, claude-3.5-sonnet, llama-3)
            use_stub: Whether to use deterministic stub instead of real API
        """
        self.model_name = model_name
        self.use_stub = use_stub
        self._used_stub_fallback = False
        
        # Check for API keys
        if not use_stub:
            if "gpt" in model_name.lower():
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    logger.warning("OPENAI_API_KEY not found, using stub")
                    self.use_stub = True
            elif "claude" in model_name.lower():
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
                if not self.api_key:
                    logger.warning("ANTHROPIC_API_KEY not found, using stub")
                    self.use_stub = True
            else:
                self.api_key = None
        
        logger.info(f"LLM Baseline initialized: {model_name} (stub={self.use_stub})")
    
    def verify_claim(
        self,
        claim: str,
        evidence: List[str],
        claim_id: str = "unknown"
    ) -> LLMPrediction:
        """
        Verify a claim using LLM judge.
        
        Args:
            claim: Claim text to verify
            evidence: List of evidence texts
            claim_id: Unique identifier
        
        Returns:
            LLMPrediction
        """
        start_time = time.time()
        inference_start = time.time()
        
        if self.use_stub:
            # Deterministic stub for testing
            result = self._stub_verify(claim, evidence)
        else:
            # Real API call
            result = self._api_verify(claim, evidence)
        
        inference_ms = (time.time() - inference_start) * 1000
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMPrediction(
            claim_id=claim_id,
            claim_text=claim,
            predicted_label=result["label"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            evidence_used=evidence[:3],  # Keep first 3 for brevity
            latency_ms=latency_ms,
            inference_ms=inference_ms
        )
    
    def _stub_verify(self, claim: str, evidence: List[str]) -> Dict:
        """
        Deterministic stub for testing without API keys.
        
        Uses heuristics to generate consistent predictions.
        """
        # Deterministic hash-based prediction
        claim_hash = hash(claim) % 100
        
        if claim_hash < 60:
            label = "SUPPORTED"
            confidence = 0.70 + (claim_hash % 30) * 0.01
        elif claim_hash < 80:
            label = "REFUTED"
            confidence = 0.65 + (claim_hash % 25) * 0.01
        else:
            label = "NEI"
            confidence = 0.50 + (claim_hash % 20) * 0.01
        
        reasoning = f"[STUB] Deterministic evaluation based on claim content. " \
                   f"Found {len(evidence)} evidence items."
        
        return {
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning
        }
    
    def _api_verify(self, claim: str, evidence: List[str]) -> Dict:
        """
        Real API verification (placeholder - implement when API keys available).
        
        This would make actual API calls to GPT-4o, Claude, etc.
        """
        # Construct prompt
        evidence_text = "\n\n".join([
            f"Evidence {i+1}: {ev[:500]}"  # Truncate long evidence
            for i, ev in enumerate(evidence[:5])  # Max 5 evidence items
        ])
        
        prompt = f"""You are an expert fact-checker. Verify the following claim using the provided evidence.

Claim: {claim}

Evidence:
{evidence_text}

Provide your assessment in JSON format:
{{
    "label": "SUPPORTED" | "REFUTED" | "NEI",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}

Consider:
- SUPPORTED: Evidence strongly supports the claim
- REFUTED: Evidence contradicts the claim
- NEI: Insufficient or conflicting evidence

Respond with ONLY the JSON."""
        
        # TODO: Implement actual API calls here
        # For now, fall back to stub
        logger.warning(f"API verification not yet implemented for {self.model_name}, using stub")
        self._used_stub_fallback = True
        return self._stub_verify(claim, evidence)
    
    def evaluate(
        self,
        test_data: List[Dict[str, Any]],
        evidence_retriever: Optional[Any] = None,
        top_k: int = 5
    ) -> LLMBaselineResult:
        """
        Evaluate LLM baseline on test data.
        
        Args:
            test_data: List of test examples with:
                - claim_id: Unique ID
                - claim_text: Claim to verify
                - true_label: Ground truth label
                - evidence (optional): Pre-retrieved evidence
            evidence_retriever: Optional evidence retriever (if evidence not provided)
        
        Returns:
            LLMBaselineResult
        """
        logger.info(f"Evaluating LLM baseline ({self.model_name}) on {len(test_data)} examples...")
        
        predictions = []
        correct_count = 0
        total_latency = 0.0
        
        # Estimate cost (rough approximation)
        tokens_per_claim = 1000  # Prompt + response
        cost_per_1k_tokens = {
            "gpt-4o": 0.005,
            "claude-3.5-sonnet": 0.003,
            "llama-3": 0.0  # Assuming local/free
        }
        cost_rate = cost_per_1k_tokens.get(self.model_name, 0.0)
        total_cost_usd = (len(test_data) * tokens_per_claim / 1000) * cost_rate
        
        for i, example in enumerate(test_data):
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(test_data)} examples...")
            
            claim_id = example.get("claim_id", f"claim_{i}")
            claim_text = example["claim_text"]
            true_label = example["true_label"]
            
            # Get evidence
            retrieval_start = time.time()

            if "evidence" in example and example["evidence"]:
                evidence = example["evidence"]
            elif evidence_retriever:
                retrieved = evidence_retriever.retrieve(claim_text, top_k=top_k)
                evidence = []
                for item in retrieved:
                    if isinstance(item, str):
                        evidence.append(item)
                    elif isinstance(item, dict):
                        evidence.append(str(item.get("text", "")))
                    else:
                        text_value = getattr(item, "text", "")
                        if text_value:
                            evidence.append(str(text_value))
            else:
                evidence = []  # No evidence

            retrieval_ms = (time.time() - retrieval_start) * 1000
            
            # Get prediction
            pred = self.verify_claim(claim_text, evidence, claim_id=claim_id)
            pred.retrieval_ms = retrieval_ms

            label_map = {
                "SUPPORTED": "VERIFIED",
                "REFUTED": "REJECTED",
                "NEI": "UNCERTAIN",
                "VERIFIED": "VERIFIED",
                "REJECTED": "REJECTED",
                "UNCERTAIN": "UNCERTAIN",
                "LOW_CONFIDENCE": "UNCERTAIN",
            }
            pred.predicted_label = label_map.get(pred.predicted_label, "UNCERTAIN")

            predictions.append(pred)
            
            # Check correctness
            if pred.predicted_label == true_label:
                correct_count += 1
            
            total_latency += pred.latency_ms
        
        # Compute metrics
        accuracy = correct_count / len(test_data)
        avg_latency_ms = total_latency / len(test_data)

        y_true = []
        y_pred = []
        confidences = []
        correctness = []
        label_to_int = {"VERIFIED": 0, "REJECTED": 1, "UNCERTAIN": 2}

        for ex, pred in zip(test_data, predictions):
            true_label = ex["true_label"]
            pred_label = pred.predicted_label
            y_true.append(label_to_int.get(true_label, 2))
            y_pred.append(label_to_int.get(pred_label, 2))
            confidences.append(float(pred.confidence))
            correctness.append(int(pred_label == true_label))

        macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if y_true else 0.0

        cal = CalibrationEvaluator(n_bins=10)
        ece = float(cal.expected_calibration_error(np.array(confidences), np.array(correctness))) if confidences else 0.0
        auc_ac = float(compute_auc_accuracy_coverage(np.array(confidences), np.array(correctness))) if confidences else 0.0

        _ = compute_risk_coverage_curve(
            scores=np.array(confidences),
            predictions=np.array(y_pred),
            targets=np.array(y_true),
            num_thresholds=100,
        )

        baseline_note = "Stub baseline (no API evaluation)" if (self.use_stub or self._used_stub_fallback) else "API-evaluated baseline"
        
        result = LLMBaselineResult(
            model_name=self.model_name,
            accuracy=accuracy,
            macro_f1=macro_f1,
            ece=ece,
            auc_ac=auc_ac,
            avg_latency_ms=avg_latency_ms,
            total_cost_usd=total_cost_usd if not self.use_stub else 0.0,
            stub_mode=self.use_stub,
            baseline_note=baseline_note,
            predictions=predictions
        )
        
        logger.info(f"LLM Baseline Results ({self.model_name}):")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Macro-F1: {macro_f1:.4f}")
        logger.info(f"  ECE: {ece:.4f}")
        logger.info(f"  AUC-AC: {auc_ac:.4f}")
        logger.info(f"  Avg Latency: {avg_latency_ms:.1f} ms")
        logger.info(f"  Total Cost: ${total_cost_usd:.2f}")
        logger.info(f"  Note: {baseline_note}")
        
        return result


def save_llm_baseline_results(
    result: LLMBaselineResult,
    output_dir: Path
):
    """
    Save LLM baseline results to files.
    
    Args:
        result: LLMBaselineResult
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        "Model": result.model_name,
        "Accuracy": result.accuracy,
        "Macro-F1": result.macro_f1,
        "ECE": result.ece,
        "AUC-AC": result.auc_ac,
        "Avg Latency (ms)": result.avg_latency_ms,
        "Total Cost (USD)": result.total_cost_usd,
        "Stub Mode": result.stub_mode,
        "Note": result.baseline_note,
    }])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame([
        {
            "Claim ID": p.claim_id,
            "Claim": p.claim_text[:100],  # Truncate
            "Predicted": p.predicted_label,
            "Confidence": p.confidence,
            "Retrieval (ms)": p.retrieval_ms,
            "Inference (ms)": p.inference_ms,
            "Latency (ms)": p.latency_ms
        }
        for p in result.predictions
    ])
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Save full JSON
    with open(output_dir / "llm_baseline_result.json", 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    logger.info(f"LLM baseline results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic test data
    test_data = []
    for i in range(20):
        test_data.append({
            "claim_id": f"claim_{i}",
            "claim_text": f"Machine learning can be used for pattern recognition task {i}.",
            "true_label": np.random.choice(["VERIFIED", "REJECTED", "UNCERTAIN"]),
            "evidence": [f"Evidence text {j} about machine learning" for j in range(3)]
        })
    
    # Evaluate with stub (no API key needed)
    baseline = LLMBaseline(model_name="gpt-4o", use_stub=True)
    result = baseline.evaluate(test_data)
    
    # Save results
    save_llm_baseline_results(result, Path("artifacts/llm_baseline_test"))
    
    print(f"\nLLM Baseline Evaluation Complete:")
    print(f"Model: {result.model_name}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Avg Latency: {result.avg_latency_ms:.1f} ms")
