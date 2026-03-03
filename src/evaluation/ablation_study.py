"""
Ablation Study for System Components.

Tests the impact of different components on overall performance:
1. Base pipeline
2. + Ensemble Confidence
3. + Temperature Scaling (Calibration)
4. + Selective Prediction

This helps understand which components contribute most to performance
and provides evidence for design decisions in the paper.
"""

import json
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment."""
    name: str
    description: str
    use_ensemble: bool = False
    use_temperature_scaling: bool = False
    use_selective_prediction: bool = False
    selective_threshold: float = 0.5
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AblationResult:
    """Results from a single ablation configuration."""
    config: AblationConfig
    accuracy: float
    macro_f1: float
    ece: float
    auc_ac: float
    latency_per_claim_ms: float = 0.0
    additional_metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        result = {
            "config": self.config.to_dict(),
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "ece": self.ece,
            "auc_ac": self.auc_ac,
            "latency_per_claim_ms": self.latency_per_claim_ms
        }
        if self.additional_metrics:
            result["additional_metrics"] = self.additional_metrics
        return result


@dataclass
class AblationStudyReport:
    """Complete ablation study report."""
    results: List[AblationResult]
    baseline_config_name: str = "Base Pipeline"
    
    def to_dict(self) -> Dict:
        return {
            "baseline_config": self.baseline_config_name,
            "results": [r.to_dict() for r in self.results]
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        rows = []
        for result in self.results:
            row = {
                "Configuration": result.config.name,
                "Accuracy": result.accuracy,
                "Macro-F1": result.macro_f1,
                "ECE (15 bins)": result.ece,
                "AUC-AC": result.auc_ac,
                "Latency (ms/claim)": result.latency_per_claim_ms
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def to_csv(self, output_path: Path):
        """Save results to CSV."""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        logger.info(f"Ablation results saved to {output_path}")
    
    def to_markdown(self, output_path: Optional[Path] = None) -> str:
        """Generate markdown table."""
        df = self.to_dataframe()
        markdown = df.to_markdown(index=False, floatfmt=".4f")
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(markdown)
            logger.info(f"Ablation markdown table saved to {output_path}")
        
        return markdown
    
    def get_improvements_over_baseline(self) -> pd.DataFrame:
        """
        Compute improvements over baseline configuration.
        
        Returns:
            DataFrame showing relative improvements for each metric
        """
        # Find baseline result
        baseline_result = None
        for result in self.results:
            if result.config.name == self.baseline_config_name:
                baseline_result = result
                break
        
        if not baseline_result:
            logger.warning(f"Baseline config '{self.baseline_config_name}' not found")
            return pd.DataFrame()
        
        # Compute improvements
        improvements = []
        for result in self.results:
            if result.config.name == self.baseline_config_name:
                continue  # Skip baseline itself
            
            improvement = {
                "Configuration": result.config.name,
                "Accuracy Δ": result.accuracy - baseline_result.accuracy,
                "Macro-F1 Δ": result.macro_f1 - baseline_result.macro_f1,
                "ECE Δ": result.ece - baseline_result.ece,  # Lower is better
                "AUC-AC Δ": result.auc_ac - baseline_result.auc_ac,
                "Latency Δ (ms)": result.latency_per_claim_ms - baseline_result.latency_per_claim_ms
            }
            improvements.append(improvement)
        
        return pd.DataFrame(improvements)


def create_default_ablation_configs() -> List[AblationConfig]:
    """
    Create default ablation study configurations.
    
    Returns:
        List of ablation configurations to test
    """
    configs = [
        AblationConfig(
            name="Base Pipeline",
            description="Raw model predictions without enhancements",
            use_ensemble=False,
            use_temperature_scaling=False,
            use_selective_prediction=False
        ),
        AblationConfig(
            name="+ Ensemble Confidence",
            description="Base + ensemble-based confidence estimation",
            use_ensemble=True,
            use_temperature_scaling=False,
            use_selective_prediction=False
        ),
        AblationConfig(
            name="+ Temperature Scaling",
            description="Base + ensemble + temperature-based calibration",
            use_ensemble=True,
            use_temperature_scaling=True,
            use_selective_prediction=False
        ),
        AblationConfig(
            name="+ Selective Prediction",
            description="Full system with selective prediction (rejection option)",
            use_ensemble=True,
            use_temperature_scaling=True,
            use_selective_prediction=True,
            selective_threshold=0.7  # Abstain below this confidence
        )
    ]
    
    return configs


def run_ablation_study(
    eval_fn: Callable[[AblationConfig], Dict[str, float]],
    configs: Optional[List[AblationConfig]] = None,
    output_dir: Optional[Path] = None
) -> AblationStudyReport:
    """
    Run ablation study across different configurations.
    
    Args:
        eval_fn: Function that takes an AblationConfig and returns metrics dict
        configs: List of configurations to test (uses defaults if None)
        output_dir: Directory to save results (optional)
    
    Returns:
        AblationStudyReport with results for all configurations
    
    Example:
        >>> def my_eval(config):
        ...     # Your evaluation code using the config
        ...     return {"accuracy": 0.85, "ece": 0.05, ...}
        >>> report = run_ablation_study(my_eval)
        >>> print(report.to_markdown())
    """
    if configs is None:
        configs = create_default_ablation_configs()
    
    logger.info(f"Running ablation study with {len(configs)} configurations...")
    
    results = []
    for config in configs:
        logger.info(f"Evaluating: {config.name}")
        logger.info(f"  Description: {config.description}")
        
        try:
            start_time = time.time()
            metrics = eval_fn(config)
            elapsed_time = time.time() - start_time
            
            result = AblationResult(
                config=config,
                accuracy=metrics.get("accuracy", 0.0),
                macro_f1=metrics.get("macro_f1", 0.0),
                ece=metrics.get("ece", 0.0),
                auc_ac=metrics.get("auc_ac", 0.0),
                latency_per_claim_ms=metrics.get("latency_per_claim_ms", elapsed_time * 1000),
                additional_metrics=metrics.get("additional_metrics")
            )
            results.append(result)
            
            logger.info(f"  Accuracy: {result.accuracy:.4f}, ECE: {result.ece:.4f}")
            
        except Exception as e:
            logger.error(f"Ablation failed for config '{config.name}': {e}")
            raise
    
    report = AblationStudyReport(results=results)
    
    # Save results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        report.to_csv(output_dir / "ablation_table.csv")
        
        # Save markdown
        report.to_markdown(output_dir / "ablation_table.md")
        
        # Save JSON
        with open(output_dir / "ablation_study.json", 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        # Save improvements over baseline
        improvements_df = report.get_improvements_over_baseline()
        if not improvements_df.empty:
            improvements_df.to_csv(output_dir / "ablation_improvements.csv", index=False)
        
        logger.info(f"Ablation study results saved to {output_dir}")
    
    # Log summary
    logger.info("\nAblation Study Summary:")
    logger.info(report.to_markdown())
    
    return report


def generate_ablation_interpretation(report: AblationStudyReport) -> str:
    """
    Generate interpretation text for ablation study results.
    
    Args:
        report: AblationStudyReport
    
    Returns:
        Markdown-formatted interpretation text
    """
    improvements = report.get_improvements_over_baseline()
    
    if improvements.empty:
        return "No improvements to analyze (baseline config not found)."
    
    interpretation_parts = []
    
    # Ensemble impact
    ensemble_rows = improvements[improvements["Configuration"].str.contains("Ensemble")]
    if not ensemble_rows.empty:
        row = ensemble_rows.iloc[0]
        acc_delta = row["Accuracy Δ"]
        ece_delta = row["ECE Δ"]
        
        interpretation_parts.append(
            f"**Ensemble Confidence**: Adding ensemble-based confidence estimation "
            f"{'improves' if acc_delta > 0 else 'reduces'} accuracy by "
            f"{abs(acc_delta):.2%} and {'reduces' if ece_delta < 0 else 'increases'} "
            f"miscalibration (ECE) by {abs(ece_delta):.4f}."
        )
    
    # Temperature scaling impact
    temp_rows = improvements[improvements["Configuration"].str.contains("Temperature")]
    if not temp_rows.empty:
        row = temp_rows.iloc[0]
        ece_delta = row["ECE Δ"]
        
        interpretation_parts.append(
            f"**Temperature Scaling**: Applying temperature-based calibration "
            f"{'further reduces' if ece_delta < 0 else 'increases'} ECE by "
            f"{abs(ece_delta):.4f}, demonstrating {'effective' if ece_delta < 0 else 'minimal'} "
            f"miscalibration correction."
        )
    
    # Selective prediction impact
    selective_rows = improvements[improvements["Configuration"].str.contains("Selective")]
    if not selective_rows.empty:
        row = selective_rows.iloc[0]
        auc_delta = row["AUC-AC Δ"]
        
        interpretation_parts.append(
            f"**Selective Prediction**: Enabling selective prediction (abstention on low confidence) "
            f"{'improves' if auc_delta > 0 else 'reduces'} AUC-AC by "
            f"{abs(auc_delta):.4f}, showing {'better' if auc_delta > 0 else 'worse'} "
            f"accuracy-coverage tradeoff."
        )
    
    return "\n\n".join(interpretation_parts)


if __name__ == "__main__":
    # Example usage
    def dummy_eval(config: AblationConfig) -> Dict[str, float]:
        """Dummy evaluation function for testing."""
        base_accuracy = 0.75
        base_ece = 0.10
        
        # Simulate improvements
        accuracy = base_accuracy
        ece = base_ece
        
        if config.use_ensemble:
            accuracy += 0.03
            ece -= 0.02
        
        if config.use_temperature_scaling:
            ece -= 0.03
        
        auc_ac = accuracy + np.random.uniform(0, 0.05)
        
        return {
            "accuracy": accuracy,
            "macro_f1": accuracy - 0.02,
            "ece": ece,
            "auc_ac": auc_ac,
            "latency_per_claim_ms": 150 + (10 if config.use_ensemble else 0)
        }
    
    report = run_ablation_study(
        eval_fn=dummy_eval,
        output_dir=Path("artifacts/ablation_test")
    )
    
    print("\n" + report.to_markdown())
    print("\n" + generate_ablation_interpretation(report))
