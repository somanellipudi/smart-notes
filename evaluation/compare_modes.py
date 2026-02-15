"""
Evaluation script comparing baseline vs verifiable mode.

Metrics:
- Hallucination rate reduction
- Confidence calibration (ECE, Brier)
- Evidence quality scores
- Graph metrics comparison
- Processing time

Outputs:
- CSV results tables
- Comparison plots
- Summary report
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Smart Notes imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning.baseline_pipeline import BaselinePipeline
from src.reasoning.verifiable_pipeline import VerifiablePipeline
from src.evaluation.calibration import CalibrationEvaluator

logger = logging.getLogger(__name__)


class ModeComparison:
    """
    Compare baseline and verifiable reasoning modes.
    
    Evaluates hallucination reduction, confidence calibration,
    and evidence quality improvements.
    """
    
    def __init__(
        self,
        output_dir: str = "outputs/evaluation",
        n_samples: int = 50
    ):
        """
        Initialize comparison evaluator.
        
        Args:
            output_dir: Directory for results
            n_samples: Number of test samples to evaluate
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_samples = n_samples
        
        self.baseline_pipeline = None
        self.verifiable_pipeline = None
        
        logger.info(f"Comparison evaluator initialized: {n_samples} samples")
    
    def initialize_pipelines(
        self,
        llm_provider,
        transcript: str,
        notes: str,
        external_context: str = ""
    ):
        """
        Initialize both baseline and verifiable pipelines.
        
        Args:
            llm_provider: LLM provider instance
            transcript: Lecture transcript
            notes: Student notes
            external_context: Additional context
        """
        logger.info("Initializing pipelines...")
        
        # Baseline (no verification)
        self.baseline_pipeline = BaselinePipeline(llm_provider)
        
        # Verifiable (with semantic + NLI verification)
        self.verifiable_pipeline = VerifiablePipeline(llm_provider)
        
        # Index sources for verifiable mode
        if hasattr(self.verifiable_pipeline, 'semantic_retriever'):
            self.verifiable_pipeline.semantic_retriever.index_sources(
                transcript=transcript,
                notes=notes,
                external_context=external_context
            )
        
        logger.info("Pipelines initialized")
    
    def run_comparison(
        self,
        test_claims: List[str],
        ground_truth_labels: Optional[List[int]] = None
    ) -> Dict:
        """
        Run both pipelines and compare results.
        
        Args:
            test_claims: List of claims to verify
            ground_truth_labels: Optional binary labels (1=correct, 0=incorrect)
        
        Returns:
            Dict with comparison metrics
        """
        results = {
            "baseline": {"claims": [], "confidences": [], "times": []},
            "verifiable": {"claims": [], "confidences": [], "times": []},
            "ground_truth": ground_truth_labels or []
        }
        
        logger.info(f"Running comparison on {len(test_claims)} claims...")
        
        # Run baseline
        for claim_text in test_claims:
            start = time.time()
            try:
                # Baseline generates claim without verification
                claim = self.baseline_pipeline.generate_claim(claim_text)
                confidence = getattr(claim, 'confidence', 0.5)
                results["baseline"]["claims"].append(claim)
                results["baseline"]["confidences"].append(confidence)
                results["baseline"]["times"].append(time.time() - start)
            except Exception as e:
                logger.error(f"Baseline failed for claim: {e}")
                results["baseline"]["claims"].append(None)
                results["baseline"]["confidences"].append(0.0)
                results["baseline"]["times"].append(time.time() - start)
        
        # Run verifiable
        for claim_text in test_claims:
            start = time.time()
            try:
                # Verifiable mode with semantic retrieval + NLI
                claim = self.verifiable_pipeline.generate_claim_with_verification(claim_text)
                confidence = getattr(claim, 'confidence', 0.5)
                results["verifiable"]["claims"].append(claim)
                results["verifiable"]["confidences"].append(confidence)
                results["verifiable"]["times"].append(time.time() - start)
            except Exception as e:
                logger.error(f"Verifiable failed for claim: {e}")
                results["verifiable"]["claims"].append(None)
                results["verifiable"]["confidences"].append(0.0)
                results["verifiable"]["times"].append(time.time() - start)
        
        logger.info("Comparison complete")
        return results
    
    def compute_metrics(
        self,
        results: Dict,
        ground_truth: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compute comparison metrics from results.
        
        Args:
            results: Output from run_comparison()
            ground_truth: Optional binary labels
        
        Returns:
            DataFrame with metrics
        """
        baseline_confs = results["baseline"]["confidences"]
        verifiable_confs = results["verifiable"]["confidences"]
        baseline_times = results["baseline"]["times"]
        verifiable_times = results["verifiable"]["times"]
        
        metrics_data = {
            "Mode": ["Baseline", "Verifiable"],
            "Avg Confidence": [
                np.mean(baseline_confs) if baseline_confs else 0.0,
                np.mean(verifiable_confs) if verifiable_confs else 0.0
            ],
            "Conf Std Dev": [
                np.std(baseline_confs) if baseline_confs else 0.0,
                np.std(verifiable_confs) if verifiable_confs else 0.0
            ],
            "Avg Time (s)": [
                np.mean(baseline_times) if baseline_times else 0.0,
                np.mean(verifiable_times) if verifiable_times else 0.0
            ],
            "Total Claims": [
                len(baseline_confs),
                len(verifiable_confs)
            ]
        }
        
        # If ground truth provided, compute calibration metrics
        if ground_truth and len(ground_truth) == len(baseline_confs):
            calibrator = CalibrationEvaluator()
            
            baseline_metrics = calibrator.evaluate(baseline_confs, ground_truth)
            verifiable_metrics = calibrator.evaluate(verifiable_confs, ground_truth)
            
            metrics_data["ECE"] = [
                baseline_metrics["ece"],
                verifiable_metrics["ece"]
            ]
            metrics_data["Brier Score"] = [
                baseline_metrics["brier_score"],
                verifiable_metrics["brier_score"]
            ]
            metrics_data["Accuracy"] = [
                baseline_metrics["accuracy"],
                verifiable_metrics["accuracy"]
            ]
            
            # Compute hallucination reduction
            baseline_hallucinations = np.sum((np.array(baseline_confs) >= 0.5) != np.array(ground_truth))
            verifiable_hallucinations = np.sum((np.array(verifiable_confs) >= 0.5) != np.array(ground_truth))
            
            reduction = ((baseline_hallucinations - verifiable_hallucinations) / 
                        max(baseline_hallucinations, 1)) * 100
            
            metrics_data["Hallucination Count"] = [
                int(baseline_hallucinations),
                int(verifiable_hallucinations)
            ]
            metrics_data["Hallucination Reduction %"] = [
                0.0,
                round(reduction, 2)
            ]
        
        df = pd.DataFrame(metrics_data)
        
        # Save to CSV
        csv_path = self.output_dir / "comparison_metrics.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Metrics saved to {csv_path}")
        
        return df
    
    def plot_confidence_distributions(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot confidence distributions for both modes.
        
        Args:
            results: Output from run_comparison()
            save_path: Path to save plot
        """
        baseline_confs = results["baseline"]["confidences"]
        verifiable_confs = results["verifiable"]["confidences"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(baseline_confs, bins=20, alpha=0.6, label='Baseline', 
               color='orange', edgecolor='black')
        ax.hist(verifiable_confs, bins=20, alpha=0.6, label='Verifiable', 
               color='blue', edgecolor='black')
        
        ax.axvline(np.mean(baseline_confs), color='orange', linestyle='--',
                  linewidth=2, label=f'Baseline mean: {np.mean(baseline_confs):.3f}')
        ax.axvline(np.mean(verifiable_confs), color='blue', linestyle='--',
                  linewidth=2, label=f'Verifiable mean: {np.mean(verifiable_confs):.3f}')
        
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Confidence Distribution: Baseline vs Verifiable', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confidence distribution plot saved to {save_path}")
        else:
            save_default = self.output_dir / "confidence_distributions.png"
            plt.savefig(save_default, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_default}")
        
        plt.close()
    
    def plot_calibration_comparison(
        self,
        results: Dict,
        ground_truth: List[int],
        save_path: Optional[str] = None
    ):
        """
        Plot reliability diagrams for both modes.
        
        Args:
            results: Output from run_comparison()
            ground_truth: Binary labels
            save_path: Path to save plot
        """
        baseline_confs = results["baseline"]["confidences"]
        verifiable_confs = results["verifiable"]["confidences"]
        
        calibrator = CalibrationEvaluator()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Baseline reliability
        self._plot_single_reliability(
            ax1, baseline_confs, ground_truth, "Baseline Mode", calibrator
        )
        
        # Verifiable reliability
        self._plot_single_reliability(
            ax2, verifiable_confs, ground_truth, "Verifiable Mode", calibrator
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Calibration comparison plot saved to {save_path}")
        else:
            save_default = self.output_dir / "calibration_comparison.png"
            plt.savefig(save_default, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_default}")
        
        plt.close()
    
    def _plot_single_reliability(
        self,
        ax,
        predictions: List[float],
        labels: List[int],
        title: str,
        calibrator: CalibrationEvaluator
    ):
        """Helper to plot reliability diagram on single axis."""
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        bin_stats = calibrator._compute_bin_statistics(predictions, labels)
        
        confidences = []
        accuracies = []
        
        for stats in bin_stats:
            if stats["count"] > 0:
                confidences.append(stats["avg_confidence"])
                accuracies.append(stats["accuracy"])
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
        ax.plot(confidences, accuracies, 'ro-', label='Model calibration', 
               linewidth=2, markersize=10)
        
        ece = calibrator.expected_calibration_error(predictions, labels)
        ax.text(0.05, 0.95, f'ECE = {ece:.4f}', fontsize=11, 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    def generate_report(
        self,
        results: Dict,
        metrics_df: pd.DataFrame,
        ground_truth: Optional[List[int]] = None
    ) -> str:
        """
        Generate summary report.
        
        Args:
            results: Comparison results
            metrics_df: Metrics DataFrame
            ground_truth: Optional labels
        
        Returns:
            Report text
        """
        report_lines = [
            "=" * 80,
            "SMART NOTES: BASELINE VS VERIFIABLE MODE COMPARISON",
            "=" * 80,
            "",
            f"Test Samples: {len(results['baseline']['confidences'])}",
            f"Ground Truth Available: {'Yes' if ground_truth else 'No'}",
            "",
            "METRICS SUMMARY:",
            "-" * 80,
            metrics_df.to_string(index=False),
            "",
            "-" * 80,
        ]
        
        if ground_truth and "Hallucination Reduction %" in metrics_df.columns:
            reduction = metrics_df.loc[1, "Hallucination Reduction %"]
            report_lines.extend([
                "",
                f"KEY FINDING: Verifiable mode reduced hallucinations by {reduction:.2f}%",
                ""
            ])
        
        report_lines.extend([
            "FILES GENERATED:",
            f"  - {self.output_dir / 'comparison_metrics.csv'}",
            f"  - {self.output_dir / 'confidence_distributions.png'}",
        ])
        
        if ground_truth:
            report_lines.append(f"  - {self.output_dir / 'calibration_comparison.png'}")
        
        report_lines.extend([
            "",
            "=" * 80
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / "comparison_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {report_path}")
        
        return report_text


def main():
    """Run comparison evaluation (demo)."""
    logging.basicConfig(level=logging.INFO)
    
    # Demo with synthetic data
    comparison = ModeComparison(output_dir="outputs/evaluation", n_samples=50)
    
    # Synthetic test claims
    test_claims = [
        "The derivative of x^2 is 2x",
        "Photosynthesis occurs in mitochondria",  # False
        "Water boils at 100°C at sea level",
        "The Earth is flat",  # False
        "Python is a programming language"
    ] * 10  # 50 claims
    
    # Synthetic ground truth (1=correct, 0=incorrect)
    ground_truth = ([1, 0, 1, 0, 1] * 10)
    
    # Synthetic results (simulate pipeline outputs)
    results = {
        "baseline": {
            "claims": [None] * 50,
            "confidences": np.random.uniform(0.4, 0.9, 50).tolist(),
            "times": np.random.uniform(1.0, 3.0, 50).tolist()
        },
        "verifiable": {
            "claims": [None] * 50,
            "confidences": np.random.uniform(0.5, 0.95, 50).tolist(),
            "times": np.random.uniform(2.0, 5.0, 50).tolist()
        },
        "ground_truth": ground_truth
    }
    
    # Compute metrics
    metrics_df = comparison.compute_metrics(results, ground_truth)
    print("\n" + metrics_df.to_string(index=False))
    
    # Generate plots
    comparison.plot_confidence_distributions(results)
    comparison.plot_calibration_comparison(results, ground_truth)
    
    # Generate report
    report = comparison.generate_report(results, metrics_df, ground_truth)
    print("\n" + report)


if __name__ == "__main__":
    main()
