"""
CS Benchmark Runner: Evaluate verification pipeline on synthetic CS dataset.

Computes:
- Accuracy, Precision, Recall per label
- Calibration: ECE (Expected Calibration Error), Brier Score
- Robustness: noise injection effects
- Efficiency: time per claim, memory usage
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import csv

from src.claims.nli_verifier import NLIVerifier, EntailmentLabel
from src.claims.validator import ClaimValidator
from src.claims.schema import LearningClaim, ClaimType, VerificationStatus, EvidenceItem
from src.retrieval.evidence_store import EvidenceStore
from src.retrieval.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Benchmark evaluation metrics."""
    
    # Classification metrics
    accuracy: float
    precision_verified: float
    recall_verified: float
    F1_verified: float
    
    precision_rejected: float
    recall_rejected: float
    F1_rejected: float
    
    precision_low_conf: float
    recall_low_conf: float
    F1_low_conf: float
    
    # Calibration metrics
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float  # Brier Score (MSE of confidence vs accuracy)
    
    # Robustness metrics
    noise_robustness_accuracy: float  # Accuracy with noise injection
    noise_types_affected: Dict[str, float]  # Noise type -> accuracy drop
    
    # Efficiency metrics
    avg_time_per_claim: float
    total_time: float
    median_time_per_claim: float
    p95_time_per_claim: float
    
    # Coverage metrics
    total_claims: int
    claims_with_evidence: int
    evidence_coverage_rate: float
    avg_evidence_count: float
    
    # Additional stats
    confidence_calibration: Dict[str, List[float]]  # Confidence bins -> accuracies
    label_distribution: Dict[str, int]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        return result
    
    def to_csv_row(self) -> Dict:
        """Convert to CSV-friendly row (flatten nested dicts)."""
        row = asdict(self)
        
        # Flatten nested dicts
        noise_types = row.pop("noise_types_affected", {})
        for noise_type, accuracy_drop in noise_types.items():
            row[f"noise_{noise_type}_drop"] = accuracy_drop
        
        confidence_cal = row.pop("confidence_calibration", {})
        for conf_bin, accuracies in confidence_cal.items():
            row[f"calibration_{conf_bin}_mean"] = np.mean(accuracies) if accuracies else 0.0
            row[f"calibration_{conf_bin}_count"] = len(accuracies)
        
        label_dist = row.pop("label_distribution", {})
        for label, count in label_dist.items():
            row[f"label_{label}_count"] = count
        
        return row


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    
    run_id: str
    timestamp: str
    dataset_path: str
    config: Dict
    metrics: BenchmarkMetrics
    predictions: List[Dict]  # {claim_id, pred_label, pred_confidence, gold_label, match}
    
    def to_csv(self, output_path: Path) -> None:
        """Save to CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics.to_csv_row().keys())
            writer.writeheader()
            writer.writerow(self.metrics.to_csv_row())
    
    def to_json(self, output_path: Path) -> None:
        """Save to JSON."""
        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "dataset_path": str(self.dataset_path),
            "config": self.config,
            "metrics": self.metrics.to_dict(),
            "predictions": self.predictions
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


class CSBenchmarkRunner:
    """Run CS verification benchmark."""
    
    def __init__(
        self,
        dataset_path: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
        nli_verifier: Optional[NLIVerifier] = None,
        batch_size: int = 8,
        device: str = "cpu",
        seed: int = 42,
        log_predictions: bool = True
    ):
        """
        Initialize benchmark runner.
        
        Args:
            dataset_path: Path to JSONL benchmark dataset
            embedding_provider: Custom embedding provider (default: all-MiniLM-L6-v2)
            nli_verifier: Custom NLI verifier (default: roberta-large-mnli)
            batch_size: Batch size for processing
            device: Device for inference ("cpu" or "cuda")
            seed: Random seed for reproducibility
            log_predictions: Whether to log individual predictions
        """
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.log_predictions = log_predictions
        
        # Set random seeds
        np.random.seed(seed)
        
        # Initialize providers
        self.embedding_provider = embedding_provider or EmbeddingProvider(device=device)
        self.nli_verifier = nli_verifier or NLIVerifier(device=device)
        
        # Load dataset
        self.dataset = self._load_dataset()
        logger.info(f"Loaded {len(self.dataset)} benchmark examples from {dataset_path}")

        # Cache for repeated runs (e.g., grid search)
        self._evidence_store = None
        self._evidence_store_size = 0
    
    def _load_dataset(self) -> List[Dict]:
        """Load JSONL dataset."""
        examples = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return examples
    
    def run(
        self,
        config: Optional[Dict] = None,
        noise_types: Optional[List[str]] = None,
        sample_size: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Run benchmark on dataset.
        
        Args:
            config: Configuration dict with flags:
                - use_retrieval: Enable retrieval (default: True)
                - use_nli: Enable NLI consistency checks (default: True)
                - use_ensemble: Use ensemble verification (default: False)
                - use_cleaning: Run text cleaning (default: True)
                - use_artifact_persistence: Save/load artifacts (default: False)
                - use_batch_nli: Use batch NLI verification (default: True)
                - use_online_authority: Query online sources (default: False)
            noise_types: Types of noise to inject: ["typo", "paraphrase", "negation", "swap"]
            sample_size: If set, use only first N examples (for CI smoke tests)
        
        Returns:
            BenchmarkResult with metrics and predictions
        """
        config = {**self._default_config(), **(config or {})}
        noise_types = noise_types or []
        
        # Use sample if specified
        dataset = self.dataset[:sample_size] if sample_size else self.dataset
        logger.info(f"Running benchmark on {len(dataset)} examples with config: {config}")
        
        # Build evidence store (reuse cache when possible)
        if self._evidence_store and self._evidence_store_size == len(dataset):
            evidence_store = self._evidence_store
            logger.info("Reusing cached evidence store...")
        else:
            logger.info("Building evidence store...")
            evidence_store = self._build_evidence_store(dataset)
            self._evidence_store = evidence_store
            self._evidence_store_size = len(dataset)
        
        # Run verification
        logger.info("Running verification pipeline...")
        predictions = []
        timings = []
        
        for example in dataset:
            start_time = time.time()
            
            # Create learning claim from example
            claim = self._example_to_claim(example, evidence_store, config)
            
            # Get prediction
            pred_label = claim.status.name if claim.status else "UNKNOWN"
            pred_confidence = claim.confidence if claim.confidence else 0.0
            
            elapsed = time.time() - start_time
            timings.append(elapsed)
            
            prediction = {
                "claim_id": example["doc_id"],
                "pred_label": pred_label,
                "pred_confidence": pred_confidence,
                "gold_label": example["gold_label"],
                "match": pred_label == example["gold_label"],
                "time": elapsed,
                "evidence_ids": [ev.evidence_id for ev in claim.evidence_objects],
                "evidence_count": len(claim.evidence_objects)
            }
            predictions.append(prediction)
            
            if self.log_predictions and prediction["claim_id"] in [d["doc_id"] for d in dataset[:5]]:
                logger.info(f"  {example['doc_id']}: {pred_label} (conf: {pred_confidence:.3f}) "
                           f"vs {example['gold_label']} [{elapsed:.3f}s]")
        
        # Compute metrics
        logger.info("Computing metrics...")
        metrics = self._compute_metrics(predictions, timings, dataset)
        
        # Test robustness if noise types specified
        if noise_types:
            logger.info(f"Testing robustness with {noise_types}...")
            metrics = self._compute_robustness(
                metrics, predictions, dataset, evidence_store, config, noise_types
            )
        
        # Build result
        run_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = BenchmarkResult(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            dataset_path=str(self.dataset_path),
            config=config,
            metrics=metrics,
            predictions=predictions
        )
        
        return result
    
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            "use_retrieval": True,
            "use_nli": True,
            "use_ensemble": False,
            "use_cleaning": True,
            "use_artifact_persistence": False,
            "use_batch_nli": True,
            "use_online_authority": False,
            "verify_threshold": 0.55,
            "low_conf_threshold": 0.35
        }
    
    def _build_evidence_store(self, dataset: List[Dict]) -> EvidenceStore:
        """Build evidence store from dataset source texts."""
        evidence_store = EvidenceStore(
            session_id=f"benchmark_{self.seed}",
            embedding_dim=384
        )
        
        # Add all source texts as evidence
        for i, example in enumerate(dataset):
            evidence_id = f"ev_{example['doc_id']}"
            source_text = example["source_text"]
            span_text = (example.get("evidence_span") or "").strip()
            if span_text and span_text in source_text:
                evidence_text = span_text
                evidence_kind = "span"
            else:
                evidence_text = source_text
                evidence_kind = "full"
            
            # Skip if already in store
            if evidence_id in evidence_store.evidence_by_id:
                continue
            
            # Create evidence object
            from src.retrieval.evidence_store import Evidence
            evidence = Evidence(
                evidence_id=evidence_id,
                source_id=example["doc_id"],
                source_type="benchmark",
                text=evidence_text,
                chunk_index=0,
                char_start=0,
                char_end=len(source_text),
                metadata={
                    "domain": example["domain_topic"],
                    "doc_id": example["doc_id"],
                    "evidence_kind": evidence_kind
                }
            )
            evidence_store.evidence.append(evidence)
            evidence_store.evidence_by_id[evidence_id] = evidence
        
        # Generate embeddings and build index
        if evidence_store.evidence:
            texts = [ev.text for ev in evidence_store.evidence]
            embeddings = self.embedding_provider.embed_texts(texts)
            
            for i, ev in enumerate(evidence_store.evidence):
                ev.embedding = embeddings[i]
            
            evidence_store.build_index(embeddings)
        
        logger.info(f"Built evidence store with {len(evidence_store.evidence)} items")
        return evidence_store
    
    def _example_to_claim(
        self,
        example: Dict,
        evidence_store: EvidenceStore,
        config: Dict
    ) -> LearningClaim:
        """Convert benchmark example to learning claim with verification."""
        claim_text = example["generated_claim"]
        
        # Create claim object
        claim = LearningClaim(
            claim_id=f"claim_{example['doc_id']}",
            claim_type=ClaimType.DEFINITION,
            claim_text=claim_text,
            metadata={"domain": example["domain_topic"]}
        )
        
        # Retrieve evidence if configured
        if config.get("use_retrieval", True):
            claim.evidence_objects = self._retrieve_evidence(claim, evidence_store)
        
        # Validate claim
        if config.get("use_nli", True):
            self._validate_claim(claim, config)
        else:
            # Simple: VERIFIED if evidence, else REJECTED
            if claim.evidence_objects:
                claim.status = VerificationStatus.VERIFIED
                claim.confidence = 0.9
            else:
                claim.status = VerificationStatus.REJECTED
                claim.confidence = 0.1
        
        return claim
    
    def _retrieve_evidence(
        self,
        claim: LearningClaim,
        evidence_store: EvidenceStore,
        top_k: int = 5
    ) -> List[EvidenceItem]:
        """Retrieve evidence for claim."""
        # Embed claim (query embedding)
        claim_embedding = self.embedding_provider.embed_queries([claim.claim_text])[0]
        
        # Search evidence store using its cosine similarity logic
        if not evidence_store.index_built:
            return []
        
        search_results = evidence_store.search(claim_embedding, top_k=top_k)
        
        evidence_items = []
        for ev, similarity in search_results:
            evidence_item = EvidenceItem(
                evidence_id=ev.evidence_id,
                source_id=ev.source_id,
                source_type=ev.source_type,
                snippet=ev.text[:200],
                span_metadata={"doc_id": ev.metadata.get("doc_id")},
                similarity=float(similarity),
                reliability_prior=0.8
            )
            evidence_items.append(evidence_item)
        
        return evidence_items
    
    def _validate_claim(self, claim: LearningClaim, config: Dict) -> None:
        """Validate claim using NLI."""
        if not claim.evidence_objects:
            claim.status = VerificationStatus.REJECTED
            claim.confidence = 0.1
            return
        
        # Use NLI to check consistency with top evidence
        top_evidence = claim.evidence_objects[:3]
        
        if config.get("use_batch_nli", True) and len(top_evidence) > 1:
            # Batch verification
            pairs = [(claim.claim_text, ev.snippet) for ev in top_evidence]
            results = self.nli_verifier.verify_batch(pairs)
            
            # Aggregate results
            entailment_scores = [r.entailment_prob for r in results]
            contradiction_scores = [r.contradiction_prob for r in results]
        else:
            # Single verification
            result = self.nli_verifier.verify(
                claim.claim_text,
                top_evidence[0].snippet
            )
            entailment_scores = [result.entailment_prob]
            contradiction_scores = [result.contradiction_prob]
        
        mean_entail = float(np.max(entailment_scores))
        max_contra = float(np.max(contradiction_scores))
        avg_sim = float(np.mean([ev.similarity for ev in claim.evidence_objects]))
        combined_score = 0.7 * mean_entail + 0.3 * avg_sim
        verify_threshold = float(config.get("verify_threshold", 0.55))
        low_conf_threshold = float(config.get("low_conf_threshold", 0.35))
        
        # Determine status and confidence
        if max_contra >= 0.6:
            claim.status = VerificationStatus.REJECTED
            claim.confidence = max(0.0, 1.0 - max_contra)
            return
        
        if combined_score >= verify_threshold and mean_entail >= 0.4:
            claim.status = VerificationStatus.VERIFIED
        elif combined_score >= low_conf_threshold:
            claim.status = VerificationStatus.LOW_CONFIDENCE
        else:
            claim.status = VerificationStatus.REJECTED
        
        claim.confidence = combined_score
    
    def _compute_metrics(
        self,
        predictions: List[Dict],
        timings: List[float],
        dataset: List[Dict]
    ) -> BenchmarkMetrics:
        """Compute benchmark metrics."""
        # Convert labels
        pred_labels = [p["pred_label"] for p in predictions]
        gold_labels = [p["gold_label"] for p in predictions]
        confidences = [p["pred_confidence"] for p in predictions]
        
        # Accuracy
        matches = [p["match"] for p in predictions]
        accuracy = np.mean(matches)
        
        # Per-label metrics
        labels = ["VERIFIED", "REJECTED", "LOW_CONFIDENCE"]
        label_metrics = {}
        
        for label in labels:
            gold_mask = np.array([g == label for g in gold_labels])
            pred_mask = np.array([p == label for p in pred_labels])
            
            tp = np.sum(gold_mask & pred_mask)
            fp = np.sum(~gold_mask & pred_mask)
            fn = np.sum(gold_mask & ~pred_mask)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            label_metrics[label] = {
                "precision": precision,
                "recall": recall,
                "F1": f1
            }
        
        # Calibration metrics (ECE and Brier score)
        ece, mce, calibration_bins = self._compute_ece(
            np.array(matches),
            np.array(confidences)
        )
        
        brier_score = np.mean((np.array(confidences) - np.array(matches)) ** 2)
        
        # Efficiency metrics
        avg_time = np.mean(timings)
        total_time = np.sum(timings)
        median_time = np.median(timings)
        p95_time = np.percentile(timings, 95)
        
        # Coverage metrics
        claims_with_evidence = sum(1 for p in predictions if p.get("evidence_count", 0) > 0)
        avg_evidence_count = sum(
            p.get("evidence_count", len(p.get("evidence_ids", []))) for p in predictions
        ) / len(predictions) if predictions else 0.0
        
        # Label distribution
        label_dist = {}
        for label in labels:
            label_dist[label] = sum(1 for g in gold_labels if g == label)
        
        return BenchmarkMetrics(
            accuracy=float(accuracy),
            precision_verified=float(label_metrics["VERIFIED"]["precision"]),
            recall_verified=float(label_metrics["VERIFIED"]["recall"]),
            F1_verified=float(label_metrics["VERIFIED"]["F1"]),
            precision_rejected=float(label_metrics["REJECTED"]["precision"]),
            recall_rejected=float(label_metrics["REJECTED"]["recall"]),
            F1_rejected=float(label_metrics["REJECTED"]["F1"]),
            precision_low_conf=float(label_metrics["LOW_CONFIDENCE"]["precision"]),
            recall_low_conf=float(label_metrics["LOW_CONFIDENCE"]["recall"]),
            F1_low_conf=float(label_metrics["LOW_CONFIDENCE"]["F1"]),
            ece=float(ece),
            mce=float(mce),
            brier_score=float(brier_score),
            noise_robustness_accuracy=1.0,  # Placeholder, computed separately
            noise_types_affected={},
            avg_time_per_claim=float(avg_time),
            total_time=float(total_time),
            median_time_per_claim=float(median_time),
            p95_time_per_claim=float(p95_time),
            total_claims=len(predictions),
            claims_with_evidence=claims_with_evidence,
            evidence_coverage_rate=claims_with_evidence / len(predictions) if predictions else 0.0,
            avg_evidence_count=float(avg_evidence_count),
            confidence_calibration=calibration_bins,
            label_distribution=label_dist
        )
    
    def _compute_ece(
        self,
        matches: np.ndarray,
        confidences: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, float, Dict]:
        """
        Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        
        ECE = average gap between confidence and accuracy in bins.
        MCE = maximum gap.
        """
        # Sort into bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        calibration = {}
        
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            
            in_bin = (confidences >= lower) & (confidences < upper)
            if np.any(in_bin):
                bin_accuracy = np.mean(matches[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                bin_size = np.sum(in_bin)
                
                calibration[f"{lower:.1f}-{upper:.1f}"] = list(matches[in_bin].astype(float))
            else:
                bin_accuracy = 0.0
                bin_confidence = 0.0
                bin_size = 0
                calibration[f"{lower:.1f}-{upper:.1f}"] = []
        
        # Compute ECE
        ece = 0.0
        mce = 0.0
        total_samples = len(matches)
        
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            
            in_bin = (confidences >= lower) & (confidences < upper)
            if np.any(in_bin):
                bin_accuracy = np.mean(matches[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                bin_size = np.sum(in_bin)
                
                gap = np.abs(bin_accuracy - bin_confidence)
                ece += (bin_size / total_samples) * gap
                mce = max(mce, gap)
        
        return ece, mce, calibration
    
    def _compute_robustness(
        self,
        metrics: BenchmarkMetrics,
        predictions: List[Dict],
        dataset: List[Dict],
        evidence_store: EvidenceStore,
        config: Dict,
        noise_types: List[str]
    ) -> BenchmarkMetrics:
        """Test robustness by injecting noise into claims."""
        noise_results = {}
        
        for noise_type in noise_types:
            noisy_predictions = []
            
            for example in dataset[:5]:  # Sample for efficiency
                noisy_claim = self._inject_noise(example["generated_claim"], noise_type)
                
                # Create claim and verify
                claim = LearningClaim(
                    claim_id=f"noisy_{example['doc_id']}_{noise_type}",
                    claim_type=ClaimType.DEFINITION,
                    claim_text=noisy_claim
                )
                
                claim.evidence_objects = self._retrieve_evidence(claim, evidence_store)
                self._validate_claim(claim, config)
                
                pred_label = claim.status.name if claim.status else "UNKNOWN"
                noisy_predictions.append({
                    "pred_label": pred_label,
                    "gold_label": example["gold_label"],
                    "match": pred_label == example["gold_label"]
                })
            
            noise_accuracy = np.mean([p["match"] for p in noisy_predictions]) if noisy_predictions else 0.0
            accuracy_drop = metrics.accuracy - noise_accuracy
            noise_results[noise_type] = float(accuracy_drop)
        
        # Update metrics
        metrics.noise_types_affected = noise_results
        metrics.noise_robustness_accuracy = 1.0 - np.mean(list(noise_results.values())) if noise_results else 1.0
        
        return metrics
    
    def _inject_noise(self, claim: str, noise_type: str) -> str:
        """Inject noise into claim for robustness testing."""
        if noise_type == "typo":
            # Replace a word character with typo
            words = claim.split()
            if words:
                idx = np.random.randint(0, len(words))
                word = words[idx]
                if len(word) > 2:
                    char_idx = np.random.randint(0, len(word))
                    words[idx] = word[:char_idx] + '?' + word[char_idx+1:]
                return " ".join(words)
        
        elif noise_type == "paraphrase":
            # Simple: truncate claim
            return claim[:len(claim)//2] + "..."
        
        elif noise_type == "negation":
            # Add negation to start
            return "It is NOT true that " + claim.lower()
        
        elif noise_type == "swap":
            # Swap first two words
            words = claim.split()
            if len(words) >= 2:
                words[0], words[1] = words[1], words[0]
                return " ".join(words)
        
        return claim
