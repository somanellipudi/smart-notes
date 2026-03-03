"""
Multi-Seed Evaluation for Model Stability Assessment.

Runs evaluation across multiple random seeds to assess:
- Mean and standard deviation of metrics
- Worst-case performance
- Seed sensitivity

This helps ensure reported results are not cherry-picked and
provides robustness evidence for the paper.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, asdict
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SeedMetrics:
    """Metrics from a single seed run."""
    seed: int
    accuracy: float
    macro_f1: float
    ece: float
    auc_ac: float
    runtime_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MultiSeedSummary:
    """Summary statistics across multiple seeds."""
    metric_name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    worst_case: float
    
    def __str__(self) -> str:
        return f"{self.metric_name}: {self.mean:.4f} ± {self.std:.4f} (worst: {self.worst_case:.4f})"


@dataclass
class MultiSeedReport:
    """Complete multi-seed evaluation report."""
    seeds: List[int]
    metrics_by_seed: List[SeedMetrics]
    summary: Dict[str, MultiSeedSummary]
    
    def to_dict(self) -> Dict:
        return {
            "seeds": self.seeds,
            "metrics_by_seed": [m.to_dict() for m in self.metrics_by_seed],
            "summary": {k: asdict(v) for k, v in self.summary.items()}
        }
    
    def to_csv(self, output_dir: Path):
        """Save results to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics by seed
        df_by_seed = pd.DataFrame([m.to_dict() for m in self.metrics_by_seed])
        df_by_seed.to_csv(output_dir / "metrics_by_seed.csv", index=False)
        
        # Save summary statistics
        summary_rows = []
        for metric_name, summary in self.summary.items():
            summary_rows.append({
                "Metric": metric_name,
                "Mean": summary.mean,
                "Std": summary.std,
                "Min": summary.min,
                "Max": summary.max,
                "Median": summary.median,
                "Worst_Case": summary.worst_case
            })
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
        
        # Save worst-case metrics
        worst_case_rows = []
        for metric_name, summary in self.summary.items():
            worst_case_rows.append({
                "Metric": metric_name,
                "Worst_Case": summary.worst_case
            })
        df_worst = pd.DataFrame(worst_case_rows)
        df_worst.to_csv(output_dir / "worst_case_metrics.csv", index=False)
        
        logger.info(f"Multi-seed results saved to {output_dir}")
    
    def summary_table_markdown(self) -> str:
        """Generate markdown summary table."""
        lines = [
            "| Metric | Mean ± Std | Worst Case |",
            "|--------|------------|------------|"
        ]
        
        for metric_name, summary in self.summary.items():
            lines.append(
                f"| {metric_name} | {summary.mean:.4f} ± {summary.std:.4f} | {summary.worst_case:.4f} |"
            )
        
        return "\n".join(lines)


def run_multi_seed_evaluation(
    eval_fn: Callable[[int], Dict[str, float]],
    seeds: List[int] = [0, 1, 2, 3, 4],
    metric_names: List[str] = ["accuracy", "macro_f1", "ece", "auc_ac"],
    higher_is_better: Optional[Dict[str, bool]] = None
) -> MultiSeedReport:
    """
    Run evaluation across multiple seeds and aggregate results.
    
    Args:
        eval_fn: Function that takes a seed and returns a dict of metrics
        seeds: List of random seeds to evaluate
        metric_names: Names of metrics to track
        higher_is_better: Dict mapping metric names to whether higher is better
                         (used to determine worst-case). Default: True for all except ECE.
    
    Returns:
        MultiSeedReport with aggregated results
    
    Example:
        >>> def my_eval(seed):
        ...     # Your evaluation code here
        ...     return {"accuracy": 0.85, "ece": 0.05}
        >>> report = run_multi_seed_evaluation(my_eval, seeds=[0, 1, 2])
        >>> print(report.summary_table_markdown())
    """
    if higher_is_better is None:
        # Default: higher is better except for ECE (calibration error)
        higher_is_better = {
            name: (name.lower() != "ece")
            for name in metric_names
        }
    
    logger.info(f"Running multi-seed evaluation with seeds: {seeds}")
    
    # Run evaluation for each seed
    results_by_seed = []
    for seed in seeds:
        logger.info(f"Evaluating seed {seed}...")
        
        try:
            metrics = eval_fn(seed)
            
            # Extract standard metrics
            seed_result = SeedMetrics(
                seed=seed,
                accuracy=metrics.get("accuracy", 0.0),
                macro_f1=metrics.get("macro_f1", 0.0),
                ece=metrics.get("ece", 0.0),
                auc_ac=metrics.get("auc_ac", 0.0),
                runtime_seconds=metrics.get("runtime_seconds", 0.0)
            )
            results_by_seed.append(seed_result)
            
        except Exception as e:
            logger.error(f"Evaluation failed for seed {seed}: {e}")
            raise
    
    # Compute summary statistics
    summary = {}
    
    for metric_name in metric_names:
        # Extract values for this metric
        values = [getattr(result, metric_name) for result in results_by_seed]
        
        # Determine worst-case (min if higher_is_better, else max)
        if higher_is_better.get(metric_name, True):
            worst_case = np.min(values)
        else:
            worst_case = np.max(values)
        
        summary[metric_name] = MultiSeedSummary(
            metric_name=metric_name,
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            median=float(np.median(values)),
            worst_case=float(worst_case)
        )
    
    report = MultiSeedReport(
        seeds=seeds,
        metrics_by_seed=results_by_seed,
        summary=summary
    )
    
    logger.info("Multi-seed evaluation complete:")
    for metric_name, summary_stats in summary.items():
        logger.info(f"  {summary_stats}")
    
    return report


def combine_seed_results(
    results_dir: Path,
    seeds: List[int],
    output_path: Optional[Path] = None
) -> MultiSeedReport:
    """
    Combine results from separate seed runs into a single report.
    
    Useful when seeds are run in parallel and results need to be aggregated.
    
    Args:
        results_dir: Directory containing per-seed result files
        seeds: List of seeds to combine
        output_path: Optional path to save combined results
    
    Returns:
        MultiSeedReport
    """
    results_dir = Path(results_dir)
    
    metrics_by_seed = []
    
    for seed in seeds:
        result_file = results_dir / f"seed_{seed}_metrics.json"
        
        if not result_file.exists():
            logger.warning(f"Missing results for seed {seed}: {result_file}")
            continue
        
        with open(result_file, 'r') as f:
            metrics = json.load(f)
        
        seed_result = SeedMetrics(
            seed=seed,
            accuracy=metrics.get("accuracy", 0.0),
            macro_f1=metrics.get("macro_f1", 0.0),
            ece=metrics.get("ece", 0.0),
            auc_ac=metrics.get("auc_ac", 0.0),
            runtime_seconds=metrics.get("runtime_seconds", 0.0)
        )
        metrics_by_seed.append(seed_result)
    
    # Compute summary statistics
    metric_names = ["accuracy", "macro_f1", "ece", "auc_ac"]
    higher_is_better = {
        "accuracy": True,
        "macro_f1": True,
        "ece": False,
        "auc_ac": True
    }
    
    summary = {}
    for metric_name in metric_names:
        values = [getattr(result, metric_name) for result in metrics_by_seed]
        
        if higher_is_better[metric_name]:
            worst_case = np.min(values)
        else:
            worst_case = np.max(values)
        
        summary[metric_name] = MultiSeedSummary(
            metric_name=metric_name,
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            median=float(np.median(values)),
            worst_case=float(worst_case)
        )
    
    report = MultiSeedReport(
        seeds=seeds,
        metrics_by_seed=metrics_by_seed,
        summary=summary
    )
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Combined report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    # Example usage
    def dummy_eval(seed: int) -> Dict[str, float]:
        """Dummy evaluation function for testing."""
        np.random.seed(seed)
        return {
            "accuracy": 0.80 + np.random.uniform(-0.02, 0.02),
            "macro_f1": 0.78 + np.random.uniform(-0.03, 0.03),
            "ece": 0.05 + np.random.uniform(-0.01, 0.01),
            "auc_ac": 0.85 + np.random.uniform(-0.02, 0.02),
            "runtime_seconds": 100 + np.random.uniform(-10, 10)
        }
    
    report = run_multi_seed_evaluation(
        eval_fn=dummy_eval,
        seeds=[0, 1, 2, 3, 4]
    )
    
    print("\n" + report.summary_table_markdown())
    
    # Save results
    output_dir = Path("artifacts/multi_seed_test")
    report.to_csv(output_dir)
