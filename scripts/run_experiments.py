"""
Batch runner for research experiments.

Runs multiple input sessions through baseline and verifiable modes,
collecting metrics for paper-ready evaluation.

Usage:
    python scripts/run_experiments.py --input-dir examples/inputs --output-dir outputs/experiments
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import csv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning.verifiable_pipeline import VerifiablePipelineWrapper
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Batch experiment runner for research evaluation."""
    
    def __init__(self, output_dir: Path):
        """Initialize experiment runner."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict[str, Any]] = []
        
        logger.info(f"ExperimentRunner initialized: output_dir={output_dir}")
    
    def load_input_session(self, filepath: Path) -> Dict[str, Any]:
        """
        Load input session JSON.
        
        Args:
            filepath: Path to input JSON
        
        Returns:
            Input session dict
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run_single_experiment(
        self,
        input_data: Dict[str, Any],
        session_id: str,
        mode: str = "verifiable"
    ) -> Dict[str, Any]:
        """
        Run a single experiment (baseline or verifiable).
        
        Args:
            input_data: Input session data
            session_id: Session identifier
            mode: "baseline" or "verifiable"
        
        Returns:
            Results dict with metrics
        """
        logger.info(f"Running {mode} experiment: {session_id}")
        
        # Extract inputs
        combined_content = input_data.get("combined_content", "")
        equations = input_data.get("equations", [])
        external_context = input_data.get("external_context", "")
        
        # Initialize pipeline
        pipeline = VerifiablePipelineWrapper(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE
        )
        
        # Run
        start_time = datetime.now()
        
        try:
            if mode == "baseline":
                output, metadata = pipeline.process(
                    combined_content=combined_content,
                    equations=equations,
                    external_context=external_context,
                    session_id=session_id,
                    verifiable_mode=False
                )
                
                result = {
                    "session_id": session_id,
                    "mode": mode,
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "total_claims": 0,
                    "verified_claims": 0,
                    "rejected_claims": 0,
                    "rejection_rate": 0.0,
                    "verification_rate": 0.0,
                    "avg_confidence": 0.0,
                    "negative_control": False,
                    "baseline_total_items": self._count_baseline_items(output.to_dict())
                }
            
            elif mode == "verifiable":
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
                    "mode": mode,
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "total_claims": metrics.get("total_claims", 0),
                    "verified_claims": metrics.get("verified_claims", 0),
                    "rejected_claims": metrics.get("rejected_claims", 0),
                    "low_confidence_claims": metrics.get("low_confidence_claims", 0),
                    "rejection_rate": metrics.get("rejection_rate", 0.0),
                    "verification_rate": metrics.get("verification_rate", 0.0),
                    "avg_confidence": metrics.get("avg_confidence", 0.0),
                    "negative_control": metrics.get("negative_control", False),
                    "total_evidence": metrics.get("negative_control_details", {}).get("total_evidence", 0),
                    "traceability_rate": metrics.get("traceability_metrics", {}).get("traceability_rate", 0.0),
                    "rejection_reasons": metrics.get("rejection_reasons", {}),
                    
                    # Ablation flags (for reproducibility)
                    "enable_evidence_first": config.ENABLE_EVIDENCE_FIRST,
                    "enable_conflict_detection": config.ENABLE_CONFLICT_DETECTION,
                    "enable_graph_confidence": config.ENABLE_GRAPH_CONFIDENCE,
                    "enable_dependency_blocking": config.ENABLE_DEPENDENCY_BLOCKING,
                    
                    # Thresholds
                    "tau": config.VERIFIABLE_RELEVANCE_THRESHOLD,
                    "k": config.VERIFIABLE_MIN_EVIDENCE,
                    "t_verify": config.VERIFIABLE_VERIFIED_THRESHOLD,
                    "t_reject": config.VERIFIABLE_REJECTED_THRESHOLD
                }
            
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            logger.info(f"Experiment completed: {session_id} ({mode})")
            return result
        
        except Exception as e:
            logger.error(f"Experiment failed: {session_id} ({mode}) - {e}")
            return {
                "session_id": session_id,
                "mode": mode,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _count_baseline_items(self, baseline_dict: Dict[str, Any]) -> int:
        """Count total items in baseline output."""
        counts = {
            "concepts": len(baseline_dict.get("key_concepts", [])),
            "equations": len(baseline_dict.get("equation_explanations", [])),
            "examples": len(baseline_dict.get("worked_examples", [])),
            "misconceptions": len(baseline_dict.get("common_mistakes", [])),
            "faqs": len(baseline_dict.get("faqs", [])),
            "connections": len(baseline_dict.get("real_world_connections", []))
        }
        return sum(counts.values())
    
    def run_batch(
        self,
        input_dir: Path,
        run_baseline: bool = True,
        run_verifiable: bool = True
    ) -> None:
        """
        Run batch experiments on all input files.
        
        Args:
            input_dir: Directory containing input JSON files
            run_baseline: Run baseline mode
            run_verifiable: Run verifiable mode
        """
        input_files = list(input_dir.glob("*.json"))
        logger.info(f"Found {len(input_files)} input files")
        
        for input_file in input_files:
            logger.info(f"Processing: {input_file.name}")
            
            input_data = self.load_input_session(input_file)
            session_base = input_file.stem
            
            if run_baseline:
                baseline_result = self.run_single_experiment(
                    input_data,
                    session_id=f"{session_base}_baseline",
                    mode="baseline"
                )
                self.results.append(baseline_result)
            
            if run_verifiable:
                verifiable_result = self.run_single_experiment(
                    input_data,
                    session_id=f"{session_base}_verifiable",
                    mode="verifiable"
                )
                self.results.append(verifiable_result)
        
        logger.info(f"Batch complete: {len(self.results)} experiments")
    
    def save_results_json(self, filename: str = "experiment_results.json") -> Path:
        """Save all results as JSON."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment_date": datetime.now().isoformat(),
                "config": {
                    "tau": config.VERIFIABLE_RELEVANCE_THRESHOLD,
                    "k": config.VERIFIABLE_MIN_EVIDENCE,
                    "t_verify": config.VERIFIABLE_VERIFIED_THRESHOLD,
                    "t_reject": config.VERIFIABLE_REJECTED_THRESHOLD,
                    "enable_evidence_first": config.ENABLE_EVIDENCE_FIRST,
                    "enable_conflict_detection": config.ENABLE_CONFLICT_DETECTION,
                    "enable_graph_confidence": config.ENABLE_GRAPH_CONFIDENCE,
                    "enable_dependency_blocking": config.ENABLE_DEPENDENCY_BLOCKING
                },
                "results": self.results
            }, f, indent=2)
        
        logger.info(f"Results saved: {filepath}")
        return filepath
    
    def save_results_csv(self, filename: str = "experiment_summary.csv") -> Path:
        """Save summary results as CSV."""
        filepath = self.output_dir / filename
        
        if not self.results:
            logger.warning("No results to save")
            return filepath
        
        fieldnames = [
            "session_id", "mode", "success", "timestamp", "processing_time",
            "total_claims", "verified_claims", "rejected_claims", "low_confidence_claims",
            "rejection_rate", "verification_rate", "avg_confidence",
            "negative_control", "total_evidence", "traceability_rate",
            "tau", "k", "t_verify", "t_reject",
            "enable_evidence_first", "enable_conflict_detection",
            "enable_graph_confidence", "enable_dependency_blocking"
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.results)
        
        logger.info(f"CSV summary saved: {filepath}")
        return filepath


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run batch experiments for research evaluation")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with input JSON files")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments", help="Output directory")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline mode")
    parser.add_argument("--no-verifiable", action="store_true", help="Skip verifiable mode")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    runner = ExperimentRunner(output_dir)
    
    runner.run_batch(
        input_dir=input_dir,
        run_baseline=not args.no_baseline,
        run_verifiable=not args.no_verifiable
    )
    
    runner.save_results_json()
    runner.save_results_csv()
    
    logger.info("Experiment run complete!")


if __name__ == "__main__":
    main()
