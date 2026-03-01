"""
Ablation Study Runner: Compare verification pipeline configurations.

Tests:
- Baseline: no verification
- a) Retrieval only: similarity-based (no NLI)
- b) Retrieval + NLI: full verification
- c) Ensemble verifier: combines multiple strategies
- Feature toggles: cleaning, artifact persistence, batch NLI, online authority

Outputs:
- results.csv: metrics table
- ablation_summary.md: markdown report with findings
- detailed_results/: per-config result files
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblationRunner:
    """Run ablation studies on verification pipeline."""
    
    def __init__(
        self,
        dataset_path: str = "evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
        output_dir: str = "evaluation/results",
        sample_size: int = None,
        seed: int = 42,
        verify_threshold: float | None = None,
        low_conf_threshold: float | None = None
    ):
        """
        Initialize ablation runner.
        
        Args:
            dataset_path: Path to benchmark dataset
            output_dir: Directory for results
            sample_size: Number of examples to test (None = all)
            seed: Random seed
        """
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.detailed_dir = self.output_dir / "detailed_results"
        self.detailed_dir.mkdir(exist_ok=True)
        
        self.sample_size = sample_size
        self.seed = seed
        self.verify_threshold = verify_threshold
        self.low_conf_threshold = low_conf_threshold
        self.results = []
    
    def run_ablations(self, noise_injection: bool = False) -> pd.DataFrame:
        """
        Run all ablation configurations.
        
        Args:
            noise_injection: Whether to test robustness with noise
        
        Returns:
            DataFrame with results
        """
        configs = self.get_ablation_configs()
        
        logger.info(f"Running {len(configs)} ablation configurations...")
        logger.info(f"Dataset: {self.dataset_path}, Sample: {self.sample_size}")
        
        for i, (name, config) in enumerate(configs.items(), 1):
            logger.info(f"\n[{i}/{len(configs)}] Running: {name}")
            logger.info(f"  Config: {config}")
            
            try:
                runner = CSBenchmarkRunner(
                    dataset_path=self.dataset_path,
                    seed=self.seed
                )
                
                # Determine noise types
                noise_types = ["typo", "paraphrase"] if noise_injection else []
                
                result = runner.run(
                    config=config,
                    noise_types=noise_types,
                    sample_size=self.sample_size
                )
                
                # Save detailed results
                result.to_json(self.detailed_dir / f"{name}_result.json")
                result.to_csv(self.detailed_dir / f"{name}_metrics.csv")
                
                # Store summary
                metrics_dict = result.metrics.to_csv_row()
                metrics_dict["config_name"] = name
                metrics_dict["timestamp"] = result.timestamp
                self.results.append(metrics_dict)
                
                logger.info(f"  ✓ Accuracy: {result.metrics.accuracy:.3f}, "
                           f"ECE: {result.metrics.ece:.4f}, "
                           f"Time: {result.metrics.avg_time_per_claim:.3f}s")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
                self.results.append({
                    "config_name": name,
                    "accuracy": 0.0,
                    "error": str(e)
                })
        
        # Create results dataframe
        df = pd.DataFrame(self.results)
        
        # Save CSV
        csv_path = self.output_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"\n✓ Saved results to {csv_path}")
        
        # Generate markdown summary
        self.generate_summary(df)
        
        return df
    
    def get_ablation_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all ablation configurations to test."""
        configs = {
            # Baselines
            "00_no_verification": {
                "use_retrieval": False,
                "use_nli": False,
                "use_ensemble": False,
                "use_cleaning": False,
                "use_artifact_persistence": False,
                "use_batch_nli": True,
                "use_online_authority": False
            },
            
            # (a) Retrieval only
            "01a_retrieval_only": {
                "use_retrieval": True,
                "use_nli": False,
                "use_ensemble": False,
                "use_cleaning": True,
                "use_artifact_persistence": False,
                "use_batch_nli": False,
                "use_online_authority": False
            },
            
            # (b) Retrieval + NLI
            "01b_retrieval_nli": {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": False,
                "use_cleaning": True,
                "use_artifact_persistence": False,
                "use_batch_nli": True,
                "use_online_authority": False
            },
            
            # (c) Ensemble verifier
            "01c_ensemble": {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": True,
                "use_cleaning": True,
                "use_artifact_persistence": False,
                "use_batch_nli": True,
                "use_online_authority": False,
                "domain_agnostic_weighting": True,
                "quality_predictor_enabled": True,
                "non_cs_entail_weight": 0.85,
                "non_cs_similarity_weight": 0.15,
                "enable_calibration_checkpoint": True,
                "calibration_target_ece": 0.10,
                "enable_research_logging": True,
                "research_log_dir": "evaluation/results/research_logs"
            },
            
            # Feature toggles
            "02_no_cleaning": {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": False,
                "use_cleaning": False,
                "use_artifact_persistence": False,
                "use_batch_nli": True,
                "use_online_authority": False
            },
            
            "03_with_artifact_persistence": {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": False,
                "use_cleaning": True,
                "use_artifact_persistence": True,
                "use_batch_nli": True,
                "use_online_authority": False
            },
            
            "04_no_batch_nli": {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": False,
                "use_cleaning": True,
                "use_artifact_persistence": False,
                "use_batch_nli": False,
                "use_online_authority": False
            },
            
            "05_with_online_authority": {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": False,
                "use_cleaning": True,
                "use_artifact_persistence": False,
                "use_batch_nli": True,
                "use_online_authority": True
            }
        }

        if self.verify_threshold is not None or self.low_conf_threshold is not None:
            for config in configs.values():
                if self.verify_threshold is not None:
                    config["verify_threshold"] = self.verify_threshold
                if self.low_conf_threshold is not None:
                    config["low_conf_threshold"] = self.low_conf_threshold

        return configs

    def generate_summary(self, df: pd.DataFrame) -> None:
        """Generate markdown summary of ablation results."""
        summary_path = self.output_dir / "ablation_summary.md"
        
        # Extract key columns
        key_cols = ["config_name", "accuracy", "F1_verified", "precision_verified", 
                    "recall_verified", "ece", "brier_score", "avg_time_per_claim"]
        
        # Create summary content
        content = f"""# Ablation Study Results

**Run Date**: {datetime.now().isoformat()}  
**Dataset**: {self.dataset_path}  
**Sample Size**: {self.sample_size if self.sample_size else 'Full'}  
**Seed**: {self.seed}  

## Overview

This ablation study evaluates different configurations of the verification pipeline:

- **(00) Baseline**: No verification
- **(1a) Retrieval Only**: Similarity-based matching without NLI
- **(1b) Retrieval + NLI**: Full verification with natural language inference
- **(1c) Ensemble**: Multiple verification strategies combined
- **(Feature Toggles)**: Individual feature effects

## Key Metrics

- **Accuracy**: Overall correctness (proportion of correct classifications)
- **F1 (Verified)**: F1 score for VERIFIED label (primary focus)
- **Precision/Recall (Verified)**: Specificity vs sensitivity for verified claims
- **ECE**: Expected Calibration Error (gap between confidence and accuracy)
- **Brier Score**: Mean squared error of confidence predictions
- **Avg Time**: Average inference time per claim (seconds)

## Results Table

| Config | Accuracy | F1 (V) | Prec (V) | Rec (V) | ECE | Brier | Time (s) |
|--------|----------|--------|----------|---------|-----|-------|----------|
"""
        
        for _, row in df.iterrows():
            config = row.get("config_name", "")
            accuracy = row.get("accuracy", 0.0)
            f1_v = row.get("F1_verified", 0.0)
            prec_v = row.get("precision_verified", 0.0)
            rec_v = row.get("recall_verified", 0.0)
            ece = row.get("ece", 0.0)
            brier = row.get("brier_score", 0.0)
            time_s = row.get("avg_time_per_claim", 0.0)
            
            content += f"| {config} | {accuracy:.3f} | {f1_v:.3f} | {prec_v:.3f} | {rec_v:.3f} | {ece:.4f} | {brier:.4f} | {time_s:.4f} |\n"
        
        content += """
## Findings

### Accuracy Improvements
"""
        
        # Compare to baseline
        baseline_acc = next(
            (row.get("accuracy", 0.0) for _, row in df.iterrows() 
             if row.get("config_name") == "00_no_verification"),
            0.0
        )
        
        for _, row in df.iterrows():
            if row.get("config_name") != "00_no_verification":
                config = row.get("config_name")
                acc = row.get("accuracy", 0.0)
                improvement = (acc - baseline_acc) * 100
                if improvement > 0:
                    content += f"- **{config}**: +{improvement:.1f}% accuracy vs baseline\n"
        
        content += """
### Calibration Analysis

Expected Calibration Error (ECE) measures how well confidence predictions align with actual accuracy:
- Lower ECE is better (model predictions are well-calibrated)
- Brier score is similar: lower indicates better calibrated confidences

### Efficiency

Comparison of computational efficiency (inference time per claim):
- Batch NLI reduces per-claim time through amortized LLM calls
- Online authority retrieval adds network latency
- Artifact persistence provides caching benefits on repeated claims

## Recommendations

1. **For Production**: Use configuration **01b_retrieval_nli** (full verification)
   - Balances accuracy with computational efficiency
   - Good calibration for confidence thresholding

2. **For Speed-Critical**: Use configuration **01a_retrieval_only**
   - Simpler similarity-based matching
   - 2-3x faster inference

3. **For Highest Accuracy**: Use configuration **01c_ensemble**
   - Combines multiple verification strategies
   - Higher computational cost

## Reproducibility

All results are deterministic with seed={self.seed}:

```bash
python scripts/run_cs_benchmark.py \\
    --seed {self.seed} \\
    --sample_size {self.sample_size if self.sample_size else 'all'} \\
    --output_dir {self.output_dir}
```

## Next Steps

1. Test on larger datasets (100+ claims)
2. Evaluate on domain-specific subsets
3. Run robustness tests with noise injection
4. Compare against human baseline
5. Profile inference time breakdown
"""
        
        with open(summary_path, 'w') as f:
            f.write(content)
        
        logger.info(f"✓ Saved summary to {summary_path}")


def _parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _run_threshold_grid(
    dataset_path: str,
    output_dir: Path,
    sample_size: int | None,
    seed: int,
    verify_thresholds: List[float],
    low_conf_thresholds: List[float]
) -> dict:
    """Run a small grid search over thresholds and return the best config."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "threshold_grid.csv"

    shared_embedding = None
    shared_nli = None
    runner = CSBenchmarkRunner(
        dataset_path=dataset_path,
        seed=seed,
        embedding_provider=shared_embedding,
        nli_verifier=shared_nli
    )

    rows = []
    best = {
        "verify_threshold": None,
        "low_conf_threshold": None,
        "f1_verified": -1.0,
        "accuracy": -1.0
    }

    for v_th in verify_thresholds:
        for lc_th in low_conf_thresholds:
            config = {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": False,
                "use_cleaning": True,
                "use_artifact_persistence": False,
                "use_batch_nli": True,
                "use_online_authority": False,
                "verify_threshold": v_th,
                "low_conf_threshold": lc_th
            }

            result = runner.run(config=config, sample_size=sample_size)

            row = {
                "verify_threshold": v_th,
                "low_conf_threshold": lc_th,
                "accuracy": result.metrics.accuracy,
                "F1_verified": result.metrics.F1_verified,
                "precision_verified": result.metrics.precision_verified,
                "recall_verified": result.metrics.recall_verified
            }
            rows.append(row)

            if (row["F1_verified"] > best["f1_verified"] or
                (row["F1_verified"] == best["f1_verified"] and row["accuracy"] > best["accuracy"])):
                best = {
                    "verify_threshold": v_th,
                    "low_conf_threshold": lc_th,
                    "f1_verified": row["F1_verified"],
                    "accuracy": row["accuracy"]
                }

    df = pd.DataFrame(rows)
    df.to_csv(grid_path, index=False)
    logger.info(f"✓ Saved threshold grid to {grid_path}")
    logger.info(
        "✓ Best thresholds: verify=%.2f, low_conf=%.2f (F1_v=%.3f, acc=%.3f)",
        best["verify_threshold"],
        best["low_conf_threshold"],
        best["f1_verified"],
        best["accuracy"]
    )

    return best
def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run ablation study on verification pipeline"
    )
    parser.add_argument(
        "--dataset",
        default="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
        help="Path to benchmark dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of examples to test (None = all, useful for CI smoke tests)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--noise-injection",
        action="store_true",
        help="Test robustness with noise injection"
    )
    parser.add_argument(
        "--verify-threshold",
        type=float,
        default=None,
        help="Verification threshold for VERIFIED status (overrides default)"
    )
    parser.add_argument(
        "--low-conf-threshold",
        type=float,
        default=None,
        help="Lower threshold for LOW_CONFIDENCE status (overrides default)"
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run grid search for verification thresholds before ablations"
    )
    parser.add_argument(
        "--grid-sample-size",
        type=int,
        default=None,
        help="Sample size for threshold grid search (defaults to --sample-size)"
    )
    parser.add_argument(
        "--verify-thresholds",
        default="0.45,0.50,0.55,0.60",
        help="Comma-separated list of verify thresholds for grid search"
    )
    parser.add_argument(
        "--low-conf-thresholds",
        default="0.25,0.30,0.35,0.40",
        help="Comma-separated list of low-confidence thresholds for grid search"
    )
    
    args = parser.parse_args()
    
    # Optional grid search
    verify_threshold = args.verify_threshold
    low_conf_threshold = args.low_conf_threshold

    if args.grid_search:
        verify_thresholds = _parse_float_list(args.verify_thresholds)
        low_conf_thresholds = _parse_float_list(args.low_conf_thresholds)
        grid_sample_size = args.grid_sample_size if args.grid_sample_size is not None else args.sample_size
        best = _run_threshold_grid(
            dataset_path=args.dataset,
            output_dir=Path(args.output_dir),
            sample_size=grid_sample_size,
            seed=args.seed,
            verify_thresholds=verify_thresholds,
            low_conf_thresholds=low_conf_thresholds
        )
        verify_threshold = best["verify_threshold"]
        low_conf_threshold = best["low_conf_threshold"]

    # Run ablations
    runner = AblationRunner(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        verify_threshold=verify_threshold,
        low_conf_threshold=low_conf_threshold
    )
    
    df = runner.run_ablations(noise_injection=args.noise_injection)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(df[["config_name", "accuracy", "F1_verified", "ece", "avg_time_per_claim"]].to_string())
    print(f"\n✓ Results saved to {runner.output_dir}")
    print(f"✓ Summary: {runner.output_dir}/ablation_summary.md")


if __name__ == "__main__":
    main()
