"""
Threshold sweep for coverage vs refusal Pareto curve.

Systematically varies tau (similarity threshold) and k (min sources)
to generate trade-off curves for paper figures.

Usage:
    python scripts/sweep_thresholds.py --input examples/inputs/example1.json --output outputs/sweep
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import csv
import itertools

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning.verifiable_pipeline import VerifiablePipelineWrapper
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThresholdSweeper:
    """Sweep tau and k for Pareto curve generation."""
    
    def __init__(self, output_dir: Path):
        """Initialize sweeper."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict[str, Any]] = []
        
        logger.info(f"ThresholdSweeper initialized: output_dir={output_dir}")
    
    def load_input(self, filepath: Path) -> Dict[str, Any]:
        """Load input JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run_sweep_point(
        self,
        input_data: Dict[str, Any],
        tau: float,
        k: int,
        t_verify: float,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Run experiment at one parameter point.
        
        Args:
            input_data: Input session
            tau: Similarity threshold
            k: Min independent sources
            t_verify: Verification confidence threshold
            session_id: Session ID
        
        Returns:
            Result dict
        """
        logger.info(f"Sweep point: tau={tau:.2f}, k={k}, t_verify={t_verify:.2f}")
        
        # Temporarily override config
        original_tau = config.VERIFIABLE_RELEVANCE_THRESHOLD
        original_k = config.VERIFIABLE_MIN_EVIDENCE
        original_t_verify = config.VERIFIABLE_VERIFIED_THRESHOLD
        
        config.VERIFIABLE_RELEVANCE_THRESHOLD = tau
        config.VERIFIABLE_MIN_EVIDENCE = k
        config.VERIFIABLE_VERIFIED_THRESHOLD = t_verify
        
        try:
            combined_content = input_data.get("combined_content", "")
            equations = input_data.get("equations", [])
            external_context = input_data.get("external_context", "")
            
            pipeline = VerifiablePipelineWrapper(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE
            )
            
            output, metadata = pipeline.process(
                combined_content=combined_content,
                equations=equations,
                external_context=external_context,
                session_id=session_id,
                verifiable_mode=True
            )
            
            metrics = metadata.get("metrics", {})
            
            result = {
                "session_id": session_id,
                "tau": tau,
                "k": k,
                "t_verify": t_verify,
                "timestamp": datetime.now().isoformat(),
                
                # Core metrics
                "total_claims": metrics.get("total_claims", 0),
                "verified_claims": metrics.get("verified_claims", 0),
                "rejected_claims": metrics.get("rejected_claims", 0),
                "low_confidence_claims": metrics.get("low_confidence_claims", 0),
                
                # Rates
                "verification_rate": metrics.get("verification_rate", 0.0),
                "rejection_rate": metrics.get("rejection_rate", 0.0),
                "refusal_rate": (
                    metrics.get("rejected_claims", 0) / metrics.get("total_claims", 1)
                    if metrics.get("total_claims", 0) > 0 else 0.0
                ),
                
                # Coverage (verified + low_conf / total)
                "coverage": (
                    (metrics.get("verified_claims", 0) + metrics.get("low_confidence_claims", 0))
                    / metrics.get("total_claims", 1)
                    if metrics.get("total_claims", 0) > 0 else 0.0
                ),
                
                # Quality
                "avg_confidence": metrics.get("avg_confidence", 0.0),
                "traceability_rate": metrics.get("traceability_metrics", {}).get("traceability_rate", 0.0),
                
                "negative_control": metrics.get("negative_control", False),
                "success": True
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Sweep point failed: tau={tau}, k={k}, t_verify={t_verify} - {e}")
            return {
                "session_id": session_id,
                "tau": tau,
                "k": k,
                "t_verify": t_verify,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        finally:
            # Restore config
            config.VERIFIABLE_RELEVANCE_THRESHOLD = original_tau
            config.VERIFIABLE_MIN_EVIDENCE = original_k
            config.VERIFIABLE_VERIFIED_THRESHOLD = original_t_verify
    
    def run_sweep(
        self,
        input_file: Path,
        tau_values: List[float],
        k_values: List[int],
        t_verify_values: List[float]
    ) -> None:
        """
        Run full parameter sweep.
        
        Args:
            input_file: Input JSON file
            tau_values: List of tau values to try
            k_values: List of k values to try
            t_verify_values: List of t_verify values to try
        """
        input_data = self.load_input(input_file)
        session_base = input_file.stem
        
        total_points = len(tau_values) * len(k_values) * len(t_verify_values)
        logger.info(f"Running sweep: {total_points} parameter combinations")
        
        point_idx = 0
        for tau, k, t_verify in itertools.product(tau_values, k_values, t_verify_values):
            point_idx += 1
            logger.info(f"Point {point_idx}/{total_points}")
            
            result = self.run_sweep_point(
                input_data=input_data,
                tau=tau,
                k=k,
                t_verify=t_verify,
                session_id=f"{session_base}_sweep_{point_idx}"
            )
            
            self.results.append(result)
        
        logger.info(f"Sweep complete: {len(self.results)} points")
    
    def save_results_json(self, filename: str = "sweep_results.json") -> Path:
        """Save results as JSON."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "sweep_date": datetime.now().isoformat(),
                "results": self.results
            }, f, indent=2)
        
        logger.info(f"Results saved: {filepath}")
        return filepath
    
    def save_results_csv(self, filename: str = "sweep_summary.csv") -> Path:
        """Save results as CSV for plotting."""
        filepath = self.output_dir / filename
        
        if not self.results:
            logger.warning("No results to save")
            return filepath
        
        fieldnames = [
            "session_id", "tau", "k", "t_verify",
            "total_claims", "verified_claims", "rejected_claims", "low_confidence_claims",
            "verification_rate", "rejection_rate", "refusal_rate", "coverage",
            "avg_confidence", "traceability_rate", "negative_control", "success"
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.results)
        
        logger.info(f"CSV saved: {filepath}")
        return filepath
    
    def plot_pareto_curve(self, filename: str = "pareto_curve.png") -> Path:
        """
        Plot coverage vs refusal Pareto curve.
        
        Args:
            filename: Output filename
        
        Returns:
            Path to saved plot
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            filepath = self.output_dir / filename
            
            # Extract data
            successful_results = [r for r in self.results if r.get("success")]
            if not successful_results:
                logger.warning("No successful results to plot")
                return filepath
            
            coverage = [r["coverage"] for r in successful_results]
            refusal = [r["refusal_rate"] for r in successful_results]
            tau_vals = [r["tau"] for r in successful_results]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scatter = ax.scatter(
                refusal, coverage,
                c=tau_vals, cmap='viridis',
                s=100, alpha=0.7, edgecolors='black'
            )
            
            ax.set_xlabel("Refusal Rate (Rejection Rate)", fontsize=12)
            ax.set_ylabel("Coverage (Verified + Low Conf)", fontsize=12)
            ax.set_title("Coverage vs Refusal Pareto Curve\n(tau sweep)", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('tau (similarity threshold)', fontsize=10)
            
            # Ideal point annotation
            ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target Coverage (0.7)')
            ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Target Refusal (0.3)')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Pareto curve saved: {filepath}")
            return filepath
        
        except ImportError:
            logger.warning("matplotlib not available; skipping plot")
            return self.output_dir / filename
        except Exception as e:
            logger.error(f"Plotting failed: {e}")
            return self.output_dir / filename


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sweep thresholds for Pareto curve")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, default="outputs/sweep", help="Output directory")
    parser.add_argument("--tau-min", type=float, default=0.1, help="Min tau value")
    parser.add_argument("--tau-max", type=float, default=0.4, help="Max tau value")
    parser.add_argument("--tau-steps", type=int, default=7, help="Number of tau steps")
    parser.add_argument("--k-values", type=str, default="1,2", help="Comma-separated k values")
    parser.add_argument("--t-verify-values", type=str, default="0.4,0.5,0.6", help="Comma-separated t_verify values")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Generate tau values
    tau_values = [
        round(args.tau_min + i * (args.tau_max - args.tau_min) / (args.tau_steps - 1), 2)
        for i in range(args.tau_steps)
    ]
    
    k_values = [int(x.strip()) for x in args.k_values.split(",")]
    t_verify_values = [float(x.strip()) for x in args.t_verify_values.split(",")]
    
    logger.info(f"Sweep configuration:")
    logger.info(f"  tau: {tau_values}")
    logger.info(f"  k: {k_values}")
    logger.info(f"  t_verify: {t_verify_values}")
    
    sweeper = ThresholdSweeper(output_dir)
    
    sweeper.run_sweep(
        input_file=input_file,
        tau_values=tau_values,
        k_values=k_values,
        t_verify_values=t_verify_values
    )
    
    sweeper.save_results_json()
    sweeper.save_results_csv()
    
    if not args.no_plot:
        sweeper.plot_pareto_curve()
    
    logger.info("Threshold sweep complete!")


if __name__ == "__main__":
    main()
